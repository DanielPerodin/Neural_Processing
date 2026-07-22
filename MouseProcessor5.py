import os
import gc
import json
import tempfile
import shutil

import numpy as np
if not hasattr(np, "in1d"):
    np.in1d = np.isin
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from neo.io import NeuralynxIO
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

import tridesclous as tdc

from tridesclous.signalpreprocessor import (
    offline_signal_preprocessor,
    estimate_medians_mads_after_preprocesing,
)

TDC_TEMP_ROOT = r"E:\FDA Raw Data\ephys\tdc_temp"

os.makedirs(
    TDC_TEMP_ROOT,
    exist_ok=True
)

# ==========================================
# TRIDESCLOUS HELPERS
# ==========================================

def get_preprocessor_params(fs, n_channels=1):
    """
    Fetch Tridesclous auto preprocessor params for the given sampling rate.
    """
    temp_folder = tempfile.mkdtemp(
        prefix="tdc_params_",
        dir=TDC_TEMP_ROOT
    )
    try:
        dummy = np.zeros((int(fs), n_channels), dtype='float32')
        dummy_file = os.path.join(temp_folder, 'dummy.raw')
        dummy.tofile(dummy_file)

        dataio = tdc.DataIO(dirname=temp_folder)
        dataio.set_data_source(
            type='RawData',
            filenames=[dummy_file],
            dtype='float32',
            sample_rate=fs,
            total_channel=n_channels
        )
        dataio.add_one_channel_group(channels=list(range(n_channels)), chan_grp=0)
        params = tdc.get_auto_params_for_catalogue(dataio, chan_grp=0)
        return params['preprocessor']
    except Exception as e:
        print(f"Warning: could not fetch TDC params, using defaults. ({e})")
        return {'highpass_freq': 300., 'lowpass_freq': 5000.,
                'smooth_size': 0, 'common_ref_removal': False}

    finally:

        dataio = None

        gc.collect()

        try:
            shutil.rmtree(
                temp_folder
            )

        except Exception as cleanup_error:

            print(
                f"Warning: could not remove "
                f"temporary parameter folder: "
                f"{cleanup_error}"
            )


def calculate_noise_metrics_excluding_spikes(
    channel_data,
    spike_indices,
    fs,
    preprocessor_params,
    snippet_duration_s=60.0,
    exclude_start_s=60.0,
    spike_mask_before_ms=1.0,
    spike_mask_after_ms=2.0,
):
    """
    Calculate MAD and RMS noise from a fixed snippet while excluding
    samples surrounding detected spikes.

    The snippet begins after exclude_start_s. Detected spike windows
    are masked before calculating the RMS noise floor.
    """

    results = {
        'mad_uV': 0.0,
        'rms_noise_uV': 1.0,
        'mean_peak_signal_uV': 0.0,
        'channel_snr': 0.0,
    }

    try:
        total_samples = channel_data.shape[0]

        snippet_samples = int(snippet_duration_s * fs)
        start = int(exclude_start_s * fs)
        stop = min(start + snippet_samples, total_samples)

        if stop <= start:
            print("  Recording is too short for requested noise snippet.")
            return results

        snippet = channel_data[start:stop]

        data_2d = (
            snippet[:, None]
            if snippet.ndim == 1
            else snippet
        )

        filtered = offline_signal_preprocessor(
            data_2d,
            fs,
            normalize=False,
            **preprocessor_params
        )

        flat = filtered[:, 0]

        _, mad = estimate_medians_mads_after_preprocesing(
            data_2d,
            fs,
            **preprocessor_params
        )

        mad_uv = float(mad[0])
        results['mad_uV'] = mad_uv

        noise_mask = np.ones(flat.shape[0], dtype=bool)

        before_samples = int(
            spike_mask_before_ms / 1000.0 * fs
        )

        after_samples = int(
            spike_mask_after_ms / 1000.0 * fs
        )

        spike_indices = np.asarray(
            spike_indices,
            dtype=np.int64
        )

        spikes_in_snippet = spike_indices[
            (spike_indices >= start)
            & (spike_indices < stop)
        ]

        relative_spikes = spikes_in_snippet - start

        for spike_index in relative_spikes:
            mask_start = max(
                0,
                spike_index - before_samples
            )

            mask_stop = min(
                flat.shape[0],
                spike_index + after_samples + 1
            )

            noise_mask[mask_start:mask_stop] = False

        noise_samples = flat[noise_mask]

        # Additional robust amplitude mask
        noise_samples = noise_samples[
            (noise_samples >= -3.0 * mad_uv)
            & (noise_samples <= 3.0 * mad_uv)
        ]

        if noise_samples.shape[0] > 0:
            results['rms_noise_uV'] = float(
                np.sqrt(
                    np.mean(noise_samples ** 2)
                )
            )

        print(
            f"  Noise snippet: "
            f"{start / fs:.2f}s -> {stop / fs:.2f}s"
        )

        print(
            f"  Spikes masked in snippet: "
            f"{len(relative_spikes)}"
        )

        print(
            f"  Noise samples retained: "
            f"{noise_samples.shape[0]:,}"
        )

    except Exception as e:
        print(f"  [noise metric error] {e}")

        import traceback
        traceback.print_exc()

    return results


def compute_snr_from_waveforms(waveforms, channel_metrics):
    """
    Fill mean_peak_signal_uV and channel_snr into channel_metrics (in-place).
        signal = |mean( min(waveform) for each spike )|
        SNR    = signal / rms_noise_uV
    """
    if waveforms is None or len(waveforms) == 0:
        channel_metrics['mean_peak_signal_uV'] = 0.0
        channel_metrics['channel_snr']         = 0.0
        return channel_metrics
    peaks     = np.array([np.min(wf) for wf in waveforms])
    mean_peak = float(abs(np.mean(peaks)))
    rms_noise = channel_metrics.get('rms_noise_uV', 1.0)
    channel_metrics['mean_peak_signal_uV'] = mean_peak
    channel_metrics['channel_snr'] = mean_peak / rms_noise if rms_noise > 0 else 0.0
    return channel_metrics

print(f"Using tridesclous version: {tdc.__version__}")

# =========================================================
# HELPER FUNCTIONS (Adapted from noiseFloor.py)
# =========================================================

def calculate_cluster_quality_metrics(waveforms, spike_times, sampling_rate, cluster_label, channel_metrics):
    """Calculate quality metrics for a specific cluster"""
    
    # Extract channel-wide metrics
    mad_uV = channel_metrics.get('mad_uV', 0)
    rms_noise_uV = channel_metrics.get('rms_noise_uV', 1)
    mean_peak_signal_uV = channel_metrics.get('mean_peak_signal_uV', 0)
    channel_snr = channel_metrics.get('channel_snr', 0)

    base_metrics = {
        "cluster_label": int(cluster_label),
        "n_spikes": len(spike_times),
        "median_peak_to_trough_amplitude": 0,
        # Channel Stats (Repeated for every cluster on this channel)
        "channel_mad_uV": float(mad_uV),
        "channel_noise_floor_rms_uV": float(rms_noise_uV),
        "channel_mean_peak_signal_uV": float(mean_peak_signal_uV),
        "channel_snr": float(channel_snr),
        "snr": float(channel_snr) # Mapped for plotting compatibility
    }

    if len(waveforms) == 0:
        return base_metrics
    
    waveforms_array = np.array(waveforms)

    # Median peak-to-trough amplitude
    peak_to_trough_amplitudes = []
    for wf in waveforms_array:
        peak_idx = np.argmin(wf)
        peak_amplitude = wf[peak_idx]
        
        if peak_idx < len(wf) - 1:
            trough_amplitude = np.max(wf[peak_idx:])
            peak_to_trough_amp = trough_amplitude - peak_amplitude
            peak_to_trough_amplitudes.append(peak_to_trough_amp)
    
    median_peak_to_trough_amplitude = np.median(peak_to_trough_amplitudes) if peak_to_trough_amplitudes else 0
    
    base_metrics.update({
        "median_peak_to_trough_amplitude": float(median_peak_to_trough_amplitude)
    })
    
    return base_metrics

def save_cluster_metrics_plot(channel_id, cluster_metrics_list, output_dir):
    """Create and save a 2x2 visualization of cluster quality metrics."""
    if not cluster_metrics_list:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Channel {channel_id} - Cluster Quality Metrics", fontsize=16, fontweight='bold')

    cluster_labels = [m['cluster_label'] for m in cluster_metrics_list]
    tick_labels    = [str(cl) for cl in cluster_labels]
    x_pos          = np.arange(len(cluster_labels))
    colors         = plt.cm.tab10(np.linspace(0, 1, max(len(cluster_labels), 1)))

    # Panel 1: Spike counts
    axes[0, 0].bar(x_pos, [m['n_spikes'] for m in cluster_metrics_list], color=colors)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(tick_labels)
    axes[0, 0].set_xlabel("Cluster Label")
    axes[0, 0].set_ylabel("Number of Spikes")
    axes[0, 0].set_title("Spike Count per Cluster")
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Channel SNR (global — same value for every cluster bar)
    axes[0, 1].bar(x_pos, [m['channel_snr'] for m in cluster_metrics_list], color=colors)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(tick_labels)
    axes[0, 1].set_xlabel("Cluster Label")
    axes[0, 1].set_ylabel("SNR (|Mean Peak| / RMS Noise)")
    axes[0, 1].set_title("Channel SNR (Global)")
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Median peak-to-trough amplitude
    axes[1, 0].bar(x_pos, [m['median_peak_to_trough_amplitude'] for m in cluster_metrics_list], color=colors)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(tick_labels)
    axes[1, 0].set_xlabel("Cluster Label")
    axes[1, 0].set_ylabel("Amplitude (µV)")
    axes[1, 0].set_title("Median Peak-to-Trough Amplitude")
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Full text summary
    axes[1, 1].axis('off')
    m0 = cluster_metrics_list[0]

    summary_text  = f"Channel {channel_id} Summary\n"
    summary_text += "=" * 34 + "\n"
    summary_text += f"MAD:                {m0['channel_mad_uV']:.3f} µV\n"
    summary_text += f"Noise Floor (RMS):  {m0['channel_noise_floor_rms_uV']:.3f} µV\n"
    summary_text += f"Signal (Mean Peak): {m0['channel_mean_peak_signal_uV']:.3f} µV\n"
    summary_text += f"Channel SNR:        {m0['channel_snr']:.3f}\n"
    summary_text += "=" * 34 + "\n\n"

    for m in cluster_metrics_list:
        summary_text += f"Cluster {m['cluster_label']}:\n"
        summary_text += f"  Spikes:              {m['n_spikes']}\n"
        summary_text += f"  Median Pk-to-Tr:     {m['median_peak_to_trough_amplitude']:.3f} µV\n\n"

    axes[1, 1].text(0.05, 0.97, summary_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontsize=9, fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.9))

    plt.tight_layout()
    filename = os.path.join(output_dir, f'channel_{channel_id}_cluster_metrics.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Cluster metrics plot saved: {filename}")
    plt.close()

def cluster_channel_with_tridesclous(channel_id, raw_data, sampling_rate, output_dir):
    """
    Run Tridesclous clustering on a single channel and save results.
    """
    print(f"\n{'='*60}")
    print(f"Processing Channel {channel_id}")
    print(f"{'='*60}")
    
    temp_folder = tempfile.mkdtemp(
        prefix=f"tdc_channel_{channel_id}_",
        dir=TDC_TEMP_ROOT
    )

    try:
        # Extract single channel data
        channel_data = raw_data[
            :,
            channel_id:channel_id+1
        ]

        if channel_data.dtype != np.float32:
            channel_data = channel_data.astype(
                np.float32,
                copy=False
            )
        total_samples = channel_data.shape[0]

        # -----------------------------------------------------------
        # STEP 1: Run spike detection FIRST so we know where the
        # spikes are before picking a snippet for the noise floor.
        # -----------------------------------------------------------
        raw_filename = os.path.join(temp_folder, 'raw_data.raw')
        channel_data.tofile(raw_filename)

        # Setup Tridesclous
        dataio = tdc.DataIO(dirname=temp_folder)
        dataio.set_data_source(type='RawData', filenames=[raw_filename],
                               dtype='float32', sample_rate=sampling_rate, total_channel=1)
        dataio.add_one_channel_group(channels=[0])

        cc = tdc.CatalogueConstructor(
            dataio=dataio,
            chan_grp=0
        )

        params = tdc.get_auto_params_for_catalogue(
            dataio,
            chan_grp=0
        )

        try:
            cc.apply_all_steps(
                params,
                verbose=True
            )

        except ValueError as e:
            if "need at least one array to concatenate" in str(e):
                print(
                    f"⚠️  Tridesclous found no usable waveform "
                    f"selection for channel {channel_id}."
                )

                print(
                    f"Skipping channel {channel_id}: "
                    "no waveforms available for clustering."
                )

                return {
                    "channel_id": channel_id,
                    "status": "NO_USABLE_WAVEFORMS",
                    "n_clusters": 0,
                    "cluster_metrics": [],
                    "channel_metrics": None
                }

            raise

        cc.make_catalogue_for_peeler()

        catalogue = dataio.load_catalogue(chan_grp=0)
        peeler = tdc.Peeler(dataio)
        peeler.change_params(catalogue=catalogue)
        peeler.run(progressbar=True)
        spikes = dataio.get_spikes(seg_num=0, chan_grp=0).copy()

        # -----------------------------------------------------------
        # STEP 2: Calculate Channel Noise Metrics (RMS SNR Method)
        # using a fixed 60s snippet after the first 60s.
        # Detected spike windows are masked before estimating noise.
        # -----------------------------------------------------------
        print("Calculating channel noise metrics (RMS method)...")

        all_spike_indices = spikes['index']

        preprocessor_params = get_preprocessor_params(
            sampling_rate,
            n_channels=1
        )

        noise_metrics = calculate_noise_metrics_excluding_spikes(
            channel_data,
            all_spike_indices,
            sampling_rate,
            preprocessor_params,
            snippet_duration_s=60.0,
            exclude_start_s=60.0,
            spike_mask_before_ms=1.0,
            spike_mask_after_ms=2.0,
        )

        channel_metrics = noise_metrics

        print(
            f"  Channel MAD: "
            f"{channel_metrics['mad_uV']:.2f} µV"
        )

        print(
            f"  Noise Floor (RMS): "
            f"{channel_metrics['rms_noise_uV']:.2f} µV"
        )

        print(
            "  Channel SNR: "
            "(pending spike extraction)"
        )
        # -----------------------------------------------------------

        # Get unique clusters (excluding noise cluster -1)
        unique_labels = np.unique(spikes['cluster_label'][spikes['cluster_label'] >= 0])
        n_clusters = len(unique_labels)
        
        n_template_samples = catalogue['centers0'].shape[1]
        pre_samples = n_template_samples // 2
        post_samples = n_template_samples - pre_samples
        
        if n_clusters == 0:
            print(f"⚠️  No clusters found for channel {channel_id}, but checking for detected spikes...")
            
            # Check for detected spikes
            all_spike_indices = spikes['index']
            
            return {
                "channel_id": channel_id,
                "status": "NO_SPIKES_DETECTED",
                "n_clusters": 0,
                "cluster_metrics": [],
                "channel_metrics": channel_metrics,
            }
        
        print(f"Found {n_clusters} clusters on channel {channel_id}")
        
        # Organize data
        cluster_data = {}
        all_waveforms = []
        all_labels = []
        
        for spike in spikes:
            peak_idx, label = spike['index'], spike['cluster_label']
            start, end = peak_idx - pre_samples, peak_idx + post_samples
            
            if start >= 0 and end <= len(channel_data) and label >= 0:
                waveform = channel_data[start:end, 0]
                spike_time = peak_idx / sampling_rate
                
                all_waveforms.append(waveform)
                all_labels.append(label)
                
                if label not in cluster_data:
                    cluster_data[label] = {'waveforms': [], 'times': []}
                
                cluster_data[label]['waveforms'].append(waveform)
                cluster_data[label]['times'].append(spike_time)
        
        # Compute SNR from all detected waveform peaks (all clusters combined)
        compute_snr_from_waveforms(np.array(all_waveforms), channel_metrics)

        # Templates and Metrics
        cluster_templates = {}
        cluster_metrics = []
        
        for label in unique_labels:
            waveforms = cluster_data[label]['waveforms']
            times = cluster_data[label]['times']
            
            mean_template = np.mean(waveforms, axis=0)
            cluster_templates[label] = mean_template
            
            metrics = calculate_cluster_quality_metrics(waveforms, times, sampling_rate, label, channel_metrics)
            cluster_metrics.append(metrics)
            
            print(f"  Cluster {label}: {len(waveforms)} spikes")
        
        # Save results
        np.save(os.path.join(output_dir, f'channel_{channel_id}_cluster_templates.npy'), cluster_templates)
        
        with open(os.path.join(output_dir, f'channel_{channel_id}_cluster_metrics.json'), 'w') as f:
            json.dump(cluster_metrics, f, indent=2)
            
        save_cluster_metrics_plot(channel_id, cluster_metrics, output_dir)
        
        # Visualizations (PCA etc)
        all_waveforms = np.array(all_waveforms)
        all_labels = np.array(all_labels)
        
        print("Performing PCA on waveforms...")

        max_pca_waveforms = 5000

        if len(all_waveforms) > max_pca_waveforms:

            rng = np.random.default_rng(42)

            pca_indices = rng.choice(
                len(all_waveforms),
                size=max_pca_waveforms,
                replace=False
            )

            pca_waveforms = all_waveforms[
                pca_indices
            ]

            pca_labels = all_labels[
                pca_indices
            ]

            print(
                f"Using {max_pca_waveforms:,} of "
                f"{len(all_waveforms):,} waveforms "
                f"for PCA visualization."
            )

        else:

            pca_waveforms = all_waveforms
            pca_labels = all_labels

        pca = PCA(
            n_components=3
        )

        principal_components = pca.fit_transform(
            pca_waveforms
        )
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        # PLOT 1: Mean Cluster Waveforms
        plt.figure(figsize=(16, 9))
        t_axis = np.linspace(-pre_samples / sampling_rate * 1000, 
                            post_samples / sampling_rate * 1000, 
                            n_template_samples)
        
        for label in unique_labels:
            mask = (all_labels == label)
            wfs_to_plot = all_waveforms[mask]
            n_to_plot = min(200, len(wfs_to_plot))
            indices_to_plot = np.random.choice(len(wfs_to_plot), n_to_plot, replace=False)
            for i in indices_to_plot:
                plt.plot(t_axis, wfs_to_plot[i], color=color_map[label], linewidth=0.5, alpha=0.15)
        
        for label in unique_labels:
            mean_template = cluster_templates[label]
            plt.plot(t_axis, mean_template, color=color_map[label], linewidth=3, zorder=10,
                    label=f"Cluster {label}")
        
        plt.title(f"Channel {channel_id} — Mean Cluster Waveforms (Ch SNR: {channel_metrics['channel_snr']:.2f})")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (µV)")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.savefig(os.path.join(output_dir, f"channel_{channel_id}_clusters_mean.png"), dpi=300)
        plt.close()
        
        # PLOT 2: PCA 2D
        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        for label in unique_labels:

            mask = (
                pca_labels == label
            )

            ax.scatter(
                principal_components[mask, 0],
                principal_components[mask, 1],
                color=color_map[label],
                s=15,
                alpha=0.7,
                label=f"Cluster {label}"
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        ax.set_title(
            f"Channel {channel_id} — PCA 2D"
        )

        ax.legend(loc='best')

        ax.grid(
            True,
            alpha=0.3,
            linestyle='--'
        )

        plt.savefig(
            os.path.join(
                output_dir,
                f"channel_{channel_id}_pca_2d.png"
            ),
            dpi=300
        )

        plt.close()
        
        # PLOT 3: 3D PCA
        fig = plt.figure(
            figsize=(12, 10)
        )

        ax = fig.add_subplot(
            111,
            projection='3d'
        )

        for label in unique_labels:

            mask = (
                pca_labels == label
            )

            ax.scatter(
                principal_components[mask, 0],
                principal_components[mask, 1],
                principal_components[mask, 2],
                color=color_map[label],
                s=15,
                alpha=0.6,
                label=f"Cluster {label}"
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

        ax.set_title(
            f"Channel {channel_id} — PCA 3D"
        )

        ax.legend(loc='best')

        plt.savefig(
            os.path.join(
                output_dir,
                f"channel_{channel_id}_pca_3d.png"
            ),
            dpi=300
        )

        plt.close()
        
        # Save spike times
        for label in unique_labels:
            df = pd.DataFrame({
                'channel_id': channel_id,
                'cluster_label': label,
                'spike_time': cluster_data[label]['times']
            })
            df.to_csv(os.path.join(output_dir, f'channel_{channel_id}_cluster_{label}_spike_times.csv'), index=False)
        
        return {
            'channel_id': channel_id,
            'status': "SUCCESS",
            'n_clusters': n_clusters,
            'cluster_templates': cluster_templates,
            'cluster_metrics': cluster_metrics,
            'channel_metrics': channel_metrics
        }

    except Exception as e:
        print(f"Error processing channel {channel_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "channel_id": channel_id,
            "status": "ERROR",
            "n_clusters": 0,
            "cluster_metrics": [],
            "channel_metrics": None
        }
    
    finally:

        try:
            peeler = None
        except:
            pass

        try:
            catalogue = None
        except:
            pass

        try:
            cc = None
        except:
            pass

        try:
            dataio = None
        except:
            pass

        try:
            spikes = None
        except:
            pass

        try:
            channel_data = None
        except:
            pass

        try:
            all_waveforms = None
        except:
            pass

        try:
            all_labels = None
        except:
            pass

        try:
            pca_waveforms = None
        except:
            pass

        try:
            pca_labels = None
        except:
            pass

        try:
            principal_components = None
        except:
            pass

        try:
            cluster_data = None
        except:
            pass

        try:
            cluster_templates = None
        except:
            pass

        gc.collect()

        try:
            shutil.rmtree(
                temp_folder
            )

        except Exception as cleanup_error:

            print(
                f"Warning: could not remove "
                f"temporary channel folder: "
                f"{cleanup_error}"
            )

def get_excluded_channels(n_channels):
    print(f"\n{'='*60}\nCHANNEL EXCLUSION\n{'='*60}")
    print(f"Total channels available: 0 to {n_channels-1}")
    response = input("\nExclude channels? (y/n): ").lower().strip()
    
    if response not in ['y', 'yes']: return []
    
    print("Enter channel IDs to exclude (comma-separated):")
    try:
        excluded = [int(ch.strip()) for ch in input("Excluded channels: ").strip().split(',')]
        return [ch for ch in excluded if 0 <= ch < n_channels]
    except Exception:
        print("Invalid input. No channels excluded.")
        return []

def generate_car_groups(
    raw_data,
    sampling_rate,
    excluded_channels,
    sliding_window=100000,
    analysis_downsample=1000,
    correlation_threshold=0.9
):
    """
    Automatically generate CAR groups from the loaded raw recording.

    Parameters
    ----------
    raw_data : np.ndarray
        Shape: samples x channels

    excluded_channels : list
        Channels excluded from CAR and clustering.

    sliding_window : int
        Moving-average window used on rectified signal.

    analysis_downsample : int
        Downsampling factor used for CAR grouping.

    correlation_threshold : float
        Minimum correlation for channels to be grouped.

    Returns
    -------
    car_groups : list[list[int]]
        Automatically generated CAR groups.

    correlation_matrix : np.ndarray
        Correlation matrix for included channels.
    """

    print("\n" + "=" * 60)
    print("AUTOMATIC PRE-CAR GROUPING")
    print("=" * 60)

    included_channels = [
        ch
        for ch in range(raw_data.shape[1])
        if ch not in excluded_channels
    ]

    if not included_channels:
        raise ValueError("No channels available for CAR grouping.")

    print(f"Included channels: {included_channels}")

    downsampled_chunks = []

    chunk_size = int(30 * sampling_rate)

    for start in range(0, raw_data.shape[0], chunk_size):

        end = min(start + chunk_size, raw_data.shape[0])

        print(
            f"Pre-CAR analysis samples "
            f"{start:,} -> {end:,}"
        )

        rectified = np.abs(
            raw_data[
                start:end,
                included_channels
            ]
        ).astype(
            np.float32,
            copy=False
        )

        moving = np.empty_like(
            rectified,
            dtype=np.float32
        )

        for ch in range(rectified.shape[1]):

            moving[:, ch] = uniform_filter1d(
                rectified[:, ch],
                size=sliding_window,
                mode="nearest"
            )

        downsampled_chunks.append(
            moving[::analysis_downsample].copy()
        )

        del rectified
        del moving

    signal_downsample = np.vstack(
        downsampled_chunks
    )

    del downsampled_chunks

    print(
        "Pre-CAR analysis array shape:",
        signal_downsample.shape
    )

    signal_sorted = np.sort(
        signal_downsample,
        axis=0
    )[::-1]

    n = max(
        1,
        round(signal_sorted.shape[0] / 100)
    )

    signal_peaks = signal_sorted[:n]

    signal_peaks_mean = np.mean(
        signal_peaks,
        axis=0
    )

    signal_peaks_mean_sorted_index = np.argsort(
        signal_peaks_mean
    )[::-1]

    correlation_matrix = np.corrcoef(
        signal_downsample,
        rowvar=False
    )

    print("Correlation matrix done.")

    high_correlation = (
        correlation_matrix > correlation_threshold
    )

    remaining = list(
        signal_peaks_mean_sorted_index
    )

    local_groups = []

    while remaining:

        anchor = remaining[0]

        current_group = [anchor]

        not_grouped = []

        for ch in remaining[1:]:

            if high_correlation[anchor, ch]:
                current_group.append(ch)

            else:
                not_grouped.append(ch)

        local_groups.append(current_group)

        remaining = not_grouped

    # Convert local included-channel indices
    # back to original channel IDs
    car_groups = []

    for group in local_groups:

        original_channel_ids = [
            int(included_channels[ch])
            for ch in group
        ]

        car_groups.append(
            original_channel_ids
        )

    print("\nGenerated CAR groups:")

    for i, group in enumerate(car_groups):
        print(
            f"  Group {i + 1}: {group}"
        )

    return car_groups, correlation_matrix

def apply_car_to_data(
    raw_data,
    car_groups,
    chunk_size=1_000_000
):
    """
    Apply CAR in-place to automatically generated CAR groups.

    CAR is applied in chunks to reduce peak memory usage.

    Groups containing only one channel are left unchanged because
    subtracting a channel from itself would zero the entire signal.
    """

    print("\nApplying CAR (in-place, chunked)...")

    total_samples = raw_data.shape[0]

    for group_index, group in enumerate(car_groups):

        print(
            f"  CAR Group {group_index + 1}: {group}"
        )

        if len(group) < 2:

            ch = group[0]

            print(
                f"    Channel {ch} is a singleton group. "
                f"Leaving unchanged."
            )

            continue

        for start in range(
            0,
            total_samples,
            chunk_size
        ):

            end = min(
                start + chunk_size,
                total_samples
            )

            print(
                f"    CAR samples "
                f"{start:,} -> {end:,}"
            )

            group_chunk = raw_data[
                start:end,
                :
            ][:, group]

            ref = group_chunk.mean(
                axis=1,
                keepdims=True
            )

            for ch in group:

                raw_data[
                    start:end,
                    ch:ch+1
                ] -= ref

            del group_chunk
            del ref

    print("CAR complete.")

    return raw_data

def save_car_comparison_plots(raw_snippet, snippet_offset, car_data, sampling_rate,
                              excluded_channels, output_dir, plot_duration_s=2.0):
    """
    Save a PNG per channel showing a short snippet of the raw (pre-CAR) and
    CAR-referenced (post-CAR) traces side by side for visual inspection.

    `raw_snippet` is a small pre-CAR array (already sliced to `plot_duration_s`
    seconds) rather than the full recording, since that's all this function
    ever needs — keeping a second full-size copy of the recording around just
    for a 2s plot is unnecessary and memory-expensive.
    `snippet_offset` is the sample index in the original recording where
    raw_snippet starts, so it lines up with the same window in car_data.
    """
    n_channels = raw_snippet.shape[1]
    included_channels = [ch for ch in range(n_channels) if ch not in excluded_channels]

    n_plot = raw_snippet.shape[0]
    start = snippet_offset
    end = snippet_offset + n_plot
    t_ms = (np.arange(n_plot) / sampling_rate) * 1000  # time axis in ms

    print(f"\nSaving pre/post-CAR comparison plots ({plot_duration_s}s snippet from middle)...")

    for ch in included_channels:
        raw_trace = raw_snippet[:, ch]
        car_trace = car_data[start:end, ch]

        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        fig.suptitle(f"Channel {ch} — Pre vs Post CAR  ({plot_duration_s}s snippet from middle)",
                     fontsize=13, fontweight='bold')

        axes[0].plot(t_ms, raw_trace, color='steelblue', linewidth=0.6)
        axes[0].set_ylabel("Amplitude (µV)")
        axes[0].set_title("Pre-CAR (raw)")
        axes[0].grid(True, alpha=0.3, linestyle='--')

        axes[1].plot(t_ms, car_trace, color='darkorange', linewidth=0.6)
        axes[1].set_ylabel("Amplitude (µV)")
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_title("Post-CAR")
        axes[1].grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        fname = os.path.join(output_dir, f"channel_{ch}_pre_post_CAR.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")


def process_all_channels(
    raw_snippet,
    snippet_offset,
    car_data,
    sampling_rate,
    excluded_channels,
    output_dir='clustering_results'
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save pre/post-CAR comparison plots before clustering
    save_car_comparison_plots(
        raw_snippet,
        snippet_offset,
        car_data,
        sampling_rate,
        excluded_channels,
        output_dir
    )

    n_channels = car_data.shape[1]

    included_channels = [
        ch
        for ch in range(n_channels)
        if ch not in excluded_channels
    ]

    print(
        f"\n{'='*60}\n"
        f"PROCESSING {len(included_channels)} CHANNELS\n"
        f"{'='*60}"
    )

    # Store summary information only.
    # Do not retain waveform/template arrays for every channel.
    all_results = {}

    for channel_id in included_channels:

        result = cluster_channel_with_tridesclous(
            channel_id,
            car_data,
            sampling_rate,
            output_dir
        )

        if result is not None:
            all_results[channel_id] = {
                'channel_id': result['channel_id'],
                'status': result['status'],
                'n_clusters': result['n_clusters'],
                'cluster_metrics': result['cluster_metrics'],
                'channel_metrics': result['channel_metrics']
            }

        # Explicitly release the complete channel result.
        del result

        gc.collect()

        print(
            f"Memory cleanup complete after "
            f"channel {channel_id}"
        )

    summary = {
        'n_channels_processed': len(all_results),
        'sampling_rate': float(sampling_rate),
        'timestamp': datetime.now().isoformat(),
        'channels': {}
    }

    for ch, res in all_results.items():

        summary['channels'][str(ch)] = {
            'status': res['status'],
            'n_clusters': res['n_clusters'],
            'cluster_metrics': res['cluster_metrics'],
            'channel_metrics': res['channel_metrics']
        }

    with open(
        os.path.join(
            output_dir,
            'processing_summary.json'
        ),
        'w'
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    return all_results

def has_csc_files(folder_path):
    try:
        files = os.listdir(folder_path)
    except Exception:
        return False

    return any(
        f.upper().startswith("CSC")
        and f.lower().endswith(".ncs")
        for f in files
    )

# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":

    parent_folder = r"E:\FDA Raw Data\ephys\mouse36"

    if not os.path.exists(parent_folder):
        parent_folder = input(
            "Enter parent folder containing session subfolders: "
        ).strip().replace('"', '')

    session_folders = sorted([
        os.path.join(parent_folder, d)
        for d in os.listdir(parent_folder)
        if (
            os.path.isdir(os.path.join(parent_folder, d))
            and not d.endswith("_Out")
            and has_csc_files(os.path.join(parent_folder, d))
        )
    ])

    if not session_folders:
        print("No session subfolders found. Exiting.")
        exit(1)

    print(f"\nFound {len(session_folders)} session(s):")

    for sf in session_folders:
        print(f"  {sf}")

    for data_folder in session_folders:

        session_name = os.path.basename(
            os.path.normpath(data_folder)
        )

        output_dir = os.path.join(
            parent_folder,
            f"{session_name}_Out"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        print(f"\n{'=' * 60}")
        print(f"SESSION: {session_name}")
        print(f"Output : {output_dir}")
        print(f"{'=' * 60}")

        try:

            reader = NeuralynxIO(
                dirname=data_folder
            )

            blk = reader.read_block(
                lazy=False
            )

            signal = (
                blk
                .segments[0]
                .analogsignals[0]
            )

            sampling_rate = (
                signal
                .sampling_rate
                .rescale("Hz")
                .item()
            )

            print(f"Original signal units: {signal.units}")

            try:
                raw_data = np.asarray(
                    signal.rescale("uV").magnitude,
                    dtype=np.float32
                )

                print("Signal converted to µV.")

            except Exception as e:
                raise ValueError(
                    f"Could not convert Neuralynx signal to µV. "
                    f"Original units: {signal.units}. Error: {e}"
                )

            del reader
            del blk
            del signal

            gc.collect()

            print(
                f"Data shape: {raw_data.shape}, "
                f"Sampling rate: {sampling_rate} Hz"
            )

            # =====================================
            # CHANNEL EXCLUSION
            # =====================================

            excluded_channels = get_excluded_channels(
                raw_data.shape[1]
            )

            # =====================================
            # AUTOMATIC PRE-CAR GROUP GENERATION
            # =====================================

            car_groups, correlation_matrix = (
                generate_car_groups(
                    raw_data,
                    sampling_rate,
                    excluded_channels,
                    sliding_window=100000,
                    analysis_downsample=1000,
                    correlation_threshold=0.9
                )
            )

            with open(
                os.path.join(
                    output_dir,
                    "car_groups.json"
                ),
                "w"
            ) as f:

                json.dump(
                    car_groups,
                    f,
                    indent=2
                )

            np.save(
                os.path.join(
                    output_dir,
                    "car_correlation_matrix.npy"
                ),
                correlation_matrix
            )

            print("Saved CAR groups.")
            print("Saved CAR correlation matrix.")

            # =====================================
            # SAVE SMALL PRE-CAR SNIPPET
            # =====================================

            plot_duration_s = 2.0

            n_plot = int(
                plot_duration_s
                * sampling_rate
            )

            mid = raw_data.shape[0] // 2

            snippet_offset = max(
                0,
                mid - n_plot // 2
            )

            snippet_end = min(
                raw_data.shape[0],
                snippet_offset + n_plot
            )

            raw_snippet = raw_data[
                snippet_offset:snippet_end
            ].copy()

            # =====================================
            # APPLY AUTOMATICALLY GENERATED CAR
            # =====================================

            car_data = apply_car_to_data(
                raw_data,
                car_groups
            )

            # =====================================
            # TRIDESCLOUS PROCESSING
            # =====================================

            process_all_channels(
                raw_snippet,
                snippet_offset,
                car_data,
                sampling_rate,
                excluded_channels,
                output_dir=output_dir
            )

            del raw_data
            del car_data
            del raw_snippet

            gc.collect()

        except Exception as e:

            print(
                f"Error processing session "
                f"{session_name}: {e}"
            )

            import traceback
            traceback.print_exc()

            continue

    print(
        "\n"
        + "=" * 60
        + "\nALL SESSIONS COMPLETE\n"
        + "=" * 60
    )