import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neo.io import NeuralynxIO
import tridesclous as tdc
# --- NEW IMPORTS FOR NOISE CALCULATION ---
from tridesclous.signalpreprocessor import (
    offline_signal_preprocessor,
    estimate_medians_mads_after_preprocesing,
)
import json
from datetime import datetime
import tempfile
import shutil
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

print(f"Using tridesclous version: {tdc.__version__}")

# =========================================================
# HELPER FUNCTIONS (Adapted from noiseFloor.py)
# =========================================================

def get_preprocessor_params(fs, n_channels=1):
    """
    Get the standard Tridesclous preprocessor params.
    """
    temp_folder = tempfile.mkdtemp(prefix="tdc_params_")
    try:
        # write a tiny dummy file just to satisfy DataIO
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
        dataio.add_one_channel_group(
            channels=list(range(n_channels)), chan_grp=0
        )
        params = tdc.get_auto_params_for_catalogue(dataio, chan_grp=0)
        return params['preprocessor']
    except Exception as e:
        print(f"Warning: Could not fetch params, using defaults. {e}")
        return {'highpass_freq': 300., 'lowpass_freq': 5000., 'smooth_size': 0, 'common_ref_removal': False}
    finally:
        try:
            shutil.rmtree(temp_folder)
        except:
            pass

def calculate_channel_noise_metrics(channel_data, fs, preprocessor_params):
    """
    Calculate Channel-wide metrics using logic from noiseFloor.py:
    1. Filter (normalize=False)
    2. MAD estimation
    3. Threshold at -3*MAD
    4. Split into Signal (spikes) and Noise (baseline)
    5. Calculate RMS for both and SNR = RMS_Signal / RMS_Noise
    """
    results = {
        'mad_uV': 0.0,
        'rms_noise_uV': 1.0,
        'rms_signal_uV': 0.0,
        'channel_snr': 0.0
    }

    try:
        # Ensure 2D array (samples, 1)
        if channel_data.ndim == 1:
            data_2d = channel_data[:, None]
        else:
            data_2d = channel_data

        # 1. Filter (keep units in uV)
        filtered = offline_signal_preprocessor(
            data_2d,
            fs,
            normalize=False,
            **preprocessor_params
        )
        flat = filtered[:, 0]

        # 2. Estimate MAD
        med, mad = estimate_medians_mads_after_preprocesing(
            data_2d,
            fs,
            **preprocessor_params
        )
        mad_uv = float(mad[0])
        results['mad_uV'] = mad_uv

        # 3. Thresholding
        threshold = -3.0 * mad_uv
        
        # Noise: Baseline (>= threshold)
        noise_samples = flat[flat >= threshold]
        
        # Signal: Spikes (< threshold)
        spike_samples = flat[flat < threshold]

        # 4. RMS Calculations
        if noise_samples.shape[0] > 0:
            results['rms_noise_uV'] = float(np.sqrt(np.mean(noise_samples ** 2)))
        
        if spike_samples.shape[0] > 0:
            results['rms_signal_uV'] = float(np.sqrt(np.mean(spike_samples ** 2)))
            
        # 5. Channel SNR
        if results['rms_noise_uV'] > 0:
            results['channel_snr'] = results['rms_signal_uV'] / results['rms_noise_uV']

    except Exception as e:
        print(f"Warning: Noise metric calculation failed: {e}")
        import traceback
        traceback.print_exc()

    return results

# =========================================================
# METRICS & PLOTTING
# =========================================================

def calculate_cluster_quality_metrics(waveforms, spike_times, sampling_rate, cluster_label, channel_metrics):
    """Calculate quality metrics for a specific cluster"""
    
    # Extract channel-wide metrics
    mad_uV = channel_metrics.get('mad_uV', 0)
    rms_noise_uV = channel_metrics.get('rms_noise_uV', 1)
    rms_signal_uV = channel_metrics.get('rms_signal_uV', 0)
    channel_snr = channel_metrics.get('channel_snr', 0)

    base_metrics = {
        "cluster_label": int(cluster_label),
        "n_spikes": len(spike_times),
        "firing_rate": 0,
        "spikes_per_minute": 0,
        "isolation_quality": 0,
        "median_peak_to_trough_amplitude": 0,
        # Channel Stats (Repeated for every cluster on this channel)
        "channel_mad_uV": float(mad_uV),
        "channel_noise_floor_rms_uV": float(rms_noise_uV),
        "channel_rms_signal_uV": float(rms_signal_uV),
        "channel_rms_snr": float(channel_snr),
        "snr": float(channel_snr) # Mapped for plotting compatibility
    }

    if len(waveforms) == 0:
        return base_metrics
    
    waveforms_array = np.array(waveforms)
    
    # Firing rate and spikes per minute
    if len(spike_times) > 1:
        firing_rate = len(spike_times) / (max(spike_times) - min(spike_times))
        spikes_per_minute = firing_rate * 60
    else:
        firing_rate = 0
        spikes_per_minute = 0
    
    # Isolation quality (coefficient of variation)
    amplitudes = [np.min(wf) for wf in waveforms_array]
    if len(amplitudes) > 1:
        isolation_quality = np.std(amplitudes) / (np.abs(np.mean(amplitudes)) + 1e-10)
    else:
        isolation_quality = 0
    
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
        "firing_rate": float(firing_rate),
        "spikes_per_minute": float(spikes_per_minute),
        "isolation_quality": float(isolation_quality),
        "median_peak_to_trough_amplitude": float(median_peak_to_trough_amplitude)
    })
    
    return base_metrics

def save_cluster_metrics_plot(channel_id, cluster_metrics_list, output_dir):
    """Create and save a visualization of cluster quality metrics"""
    if not cluster_metrics_list:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Channel {channel_id} - Cluster Quality Metrics", fontsize=16, fontweight='bold')
    
    cluster_labels = [m['cluster_label'] for m in cluster_metrics_list]
    colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_labels)))
    
    # Plot 1: Spike counts
    axes[0, 0].bar(cluster_labels, [m['n_spikes'] for m in cluster_metrics_list], color=colors)
    axes[0, 0].set_xlabel("Cluster Label")
    axes[0, 0].set_ylabel("Number of Spikes")
    axes[0, 0].set_title("Spike Count per Cluster")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Firing rates
    axes[0, 1].bar(cluster_labels, [m['firing_rate'] for m in cluster_metrics_list], color=colors)
    axes[0, 1].set_xlabel("Cluster Label")
    axes[0, 1].set_ylabel("Firing Rate (Hz)")
    axes[0, 1].set_title("Firing Rate per Cluster")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Channel SNR (RMS) - Identical for all clusters
    axes[0, 2].bar(cluster_labels, [m['channel_rms_snr'] for m in cluster_metrics_list], color=colors)
    axes[0, 2].set_xlabel("Cluster Label")
    axes[0, 2].set_ylabel("SNR (RMS Signal / RMS Noise)")
    axes[0, 2].set_title("Channel SNR (Global)")
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Isolation quality
    axes[1, 0].bar(cluster_labels, [m['isolation_quality'] for m in cluster_metrics_list], color=colors)
    axes[1, 0].set_xlabel("Cluster Label")
    axes[1, 0].set_ylabel("Isolation Quality (CV)")
    axes[1, 0].set_title("Isolation Quality")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Peak-to-trough amplitude
    axes[1, 1].bar(cluster_labels, [m['median_peak_to_trough_amplitude'] for m in cluster_metrics_list], color=colors)
    axes[1, 1].set_xlabel("Cluster Label")
    axes[1, 1].set_ylabel("Amplitude (µV)")
    axes[1, 1].set_title("Median Peak-to-Trough Amplitude")
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Text summary
    axes[1, 2].axis('off')
    
    # Extract channel stats (same for all clusters)
    m0 = cluster_metrics_list[0]
    
    summary_text = f"Channel {channel_id} Summary\n"
    summary_text += "="*30 + "\n"
    summary_text += f"Channel MAD:       {m0['channel_mad_uV']:.2f} µV\n"
    summary_text += f"Noise Floor (RMS): {m0['channel_noise_floor_rms_uV']:.2f} µV\n"
    summary_text += f"Signal (RMS):      {m0['channel_rms_signal_uV']:.2f} µV\n"
    summary_text += f"Channel SNR:       {m0['channel_rms_snr']:.2f}\n"
    summary_text += "="*30 + "\n\n"
    
    for m in cluster_metrics_list:
        summary_text += f"Cluster {m['cluster_label']}:\n"
        summary_text += f"  Spikes: {m['n_spikes']}\n"
        summary_text += f"  Rate: {m['firing_rate']:.2f} Hz\n\n"
    
    axes[1, 2].text(0.1, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    verticalalignment='top', fontsize=10, fontfamily='monospace',
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
    
    temp_folder = tempfile.mkdtemp(prefix=f"tdc_channel_{channel_id}_")

    try:
        # Extract single channel data
        channel_data = raw_data[:, channel_id:channel_id+1].astype('float32')
        
        # -----------------------------------------------------------
        # STEP 1: Calculate Channel Noise Metrics (RMS SNR Method)
        # -----------------------------------------------------------
        print("Calculating channel noise metrics (RMS method)...")
        
        # We use a 60s snippet from the middle for consistency with noiseFloor.py
        # or the whole file if it's small.
        total_samples = channel_data.shape[0]
        snippet_len = int(60 * sampling_rate)
        
        if total_samples > snippet_len:
            mid = total_samples // 2
            start = mid - (snippet_len // 2)
            end = start + snippet_len
            metric_data = channel_data[start:end]
        else:
            metric_data = channel_data
            
        tdc_params = get_preprocessor_params(sampling_rate)
        channel_metrics = calculate_channel_noise_metrics(metric_data, sampling_rate, tdc_params)
        
        print(f"  Channel MAD: {channel_metrics['mad_uV']:.2f} µV")
        print(f"  Noise Floor (RMS): {channel_metrics['rms_noise_uV']:.2f} µV")
        print(f"  Channel RMS SNR: {channel_metrics['channel_snr']:.2f}")
        # -----------------------------------------------------------

        # Save raw data for TDC
        raw_filename = os.path.join(temp_folder, 'raw_data.raw')
        channel_data.tofile(raw_filename)

        # Setup Tridesclous
        dataio = tdc.DataIO(dirname=temp_folder)
        dataio.set_data_source(type='RawData', filenames=[raw_filename],
                               dtype='float32', sample_rate=sampling_rate, total_channel=1)
        dataio.add_one_channel_group(channels=[0])

        cc = tdc.CatalogueConstructor(dataio=dataio, chan_grp=0)
        params = tdc.get_auto_params_for_catalogue(dataio, chan_grp=0)
        cc.apply_all_steps(params, verbose=True)
        cc.make_catalogue_for_peeler()

        catalogue = dataio.load_catalogue(chan_grp=0)
        peeler = tdc.Peeler(dataio)
        peeler.change_params(catalogue=catalogue)
        peeler.run(progressbar=True)
        spikes = dataio.get_spikes(seg_num=0, chan_grp=0).copy()

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
            
            if len(all_spike_indices) == 0:
                print(f"No spikes detected at all for channel {channel_id}")
                return None
            
            print(f"Found {len(all_spike_indices)} detected spikes (unclustered)")
            
            # Extract waveforms
            all_waveforms = []
            all_spike_times = []
            
            for spike in spikes:
                peak_idx = spike['index']
                start, end = peak_idx - pre_samples, peak_idx + post_samples
                
                if start >= 0 and end <= len(channel_data):
                    waveform = channel_data[start:end, 0]
                    spike_time = peak_idx / sampling_rate
                    all_waveforms.append(waveform)
                    all_spike_times.append(spike_time)
            
            if len(all_waveforms) == 0:
                return None
            
            all_waveforms = np.array(all_waveforms)
            mean_template = np.mean(all_waveforms, axis=0)
            
            # Save "no cluster" template
            templates_file = os.path.join(output_dir, f'channel_{channel_id}_NO_CLUSTERS_detected_spikes_template.npy')
            np.save(templates_file, {'no_cluster_template': mean_template})
            
            # Metrics (Passing Channel Metrics)
            metrics = calculate_cluster_quality_metrics(all_waveforms, all_spike_times, sampling_rate, -1, channel_metrics)
            metrics['cluster_label'] = 'NO_CLUSTERS'
            
            metrics_file = os.path.join(output_dir, f'channel_{channel_id}_NO_CLUSTERS_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Visualization
            t_axis = np.linspace(-pre_samples / sampling_rate * 1000, 
                                post_samples / sampling_rate * 1000, 
                                n_template_samples)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"Channel {channel_id} — NO CLUSTERS FOUND ({len(all_waveforms)} spikes detected)", 
                        fontsize=16, fontweight='bold', color='orange')
            
            # Plot 1: Waveforms
            ax = axes[0, 0]
            n_to_plot = min(200, len(all_waveforms))
            indices_to_plot = np.random.choice(len(all_waveforms), n_to_plot, replace=False)
            for i in indices_to_plot:
                ax.plot(t_axis, all_waveforms[i], color='lightgray', linewidth=0.5, alpha=0.3)
            ax.plot(t_axis, mean_template, color='red', linewidth=3, label='Mean Template')
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Spike Hist
            ax = axes[0, 1]
            ax.hist(all_spike_times, bins=min(50, len(all_spike_times)//5 + 1), alpha=0.7, color='gray')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: PCA
            ax = axes[1, 0]
            if len(all_waveforms) >= 3:
                try:
                    pca = PCA(n_components=min(2, len(all_waveforms)))
                    pca_2d = pca.fit_transform(all_waveforms)
                    ax.scatter(pca_2d[:, 0], pca_2d[:, 1], color='gray', s=10, alpha=0.5)
                except:
                    ax.text(0.5, 0.5, "PCA failed", ha='center')
            
            # Plot 4: Metrics Text
            ax = axes[1, 1]
            ax.axis('off')
            metrics_text = f"Channel {channel_id} — NO CLUSTERS\n"
            metrics_text += "="*40 + "\n\n"
            metrics_text += f"Channel MAD: {metrics['channel_mad_uV']:.2f} µV\n"
            metrics_text += f"Noise Floor (RMS): {metrics['channel_noise_floor_rms_uV']:.2f} µV\n"
            metrics_text += f"Channel SNR: {metrics['channel_rms_snr']:.2f}\n"
            metrics_text += "-"*40 + "\n"
            metrics_text += f"Total Spikes: {len(all_waveforms)}\n"
            metrics_text += f"Firing Rate: {metrics['firing_rate']:.2f} Hz\n"
            
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.9), fontfamily='monospace')
            
            plt.savefig(os.path.join(output_dir, f"channel_{channel_id}_NO_CLUSTERS.png"), dpi=300)
            plt.close()
            
            return {
                'channel_id': channel_id,
                'n_clusters': 0,
                'n_spikes_detected': len(all_waveforms),
                'cluster_metrics': [metrics],
                'status': 'NO_CLUSTERS_DETECTED',
                'channel_metrics': channel_metrics
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
        
        # Templates and Metrics
        cluster_templates = {}
        cluster_metrics = []
        
        for label in unique_labels:
            waveforms = cluster_data[label]['waveforms']
            times = cluster_data[label]['times']
            
            mean_template = np.mean(waveforms, axis=0)
            cluster_templates[label] = mean_template
            
            # Passing channel_metrics for RMS SNR
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
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(all_waveforms)
        
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
        plt.legend(loc='best')
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.savefig(os.path.join(output_dir, f"channel_{channel_id}_clusters_mean.png"), dpi=300)
        plt.close()
        
        # PLOT 2: PCA
        fig, ax = plt.subplots(figsize=(10, 8))
        for label in unique_labels:
            mask = (all_labels == label)
            ax.scatter(principal_components[mask, 0], principal_components[mask, 1],
                      color=color_map[label], s=15, alpha=0.7, label=f"Cluster {label}")
        plt.savefig(os.path.join(output_dir, f"channel_{channel_id}_pca_2d.png"), dpi=300)
        plt.close()
        
        # PLOT 3: 3D PCA
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for label in unique_labels:
            mask = (all_labels == label)
            ax.scatter(principal_components[mask, 0], principal_components[mask, 1], 
                      principal_components[mask, 2],
                      color=color_map[label], s=15, alpha=0.6)
        plt.savefig(os.path.join(output_dir, f"channel_{channel_id}_pca_3d.png"), dpi=300)
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
            'n_clusters': n_clusters,
            'cluster_templates': cluster_templates,
            'cluster_metrics': cluster_metrics,
            'channel_metrics': channel_metrics
        }

    except Exception as e:
        print(f"Error processing channel {channel_id}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        try:
            shutil.rmtree(temp_folder)
        except:
            pass

def get_excluded_channels(n_channels):
    print(f"\n{'='*60}\nCHANNEL EXCLUSION\n{'='*60}")
    print(f"Total channels available: 0 to {n_channels-1}")
    response = input("\nExclude channels? (y/n): ").lower().strip()
    
    if response not in ['y', 'yes']: return []
    
    print("Enter channel IDs to exclude (comma-separated):")
    try:
        excluded = [int(ch.strip()) for ch in input("Excluded channels: ").strip().split(',')]
        return [ch for ch in excluded if 0 <= ch < n_channels]
    except:
        return []

def get_car_groups(n_channels, excluded_channels):
    included_channels = [ch for ch in range(n_channels) if ch not in excluded_channels]
    print(f"\n{'='*60}\nCAR GROUPING\n{'='*60}")
    print("1. All included channels (default)\n2. Custom groups\n3. No CAR")
    
    response = input("Choice (1/2/3): ").strip()
    
    if response == '2':
        print("Enter groups (comma-separated, one per line). Empty line to finish.")
        groups = []
        while True:
            line = input(f"Group {len(groups)+1}: ").strip()
            if not line: break
            try:
                g = [int(ch) for ch in line.split(',') if int(ch) in included_channels]
                if g: groups.append(g)
            except: pass
        return groups if groups else [included_channels]
    elif response == '3':
        return [[ch] for ch in included_channels]
    else:
        return [included_channels]

def apply_car_to_data(raw_data, car_groups):
    car_data = raw_data.copy()
    print("\nApplying CAR...")
    for group in car_groups:
        if len(group) == 1:
            ch = group[0]
            car_data[:, ch:ch+1] -= raw_data[:, ch:ch+1].mean(axis=1, keepdims=True)
        else:
            ref = raw_data[:, group].mean(axis=1, keepdims=True)
            for ch in group:
                car_data[:, ch:ch+1] -= ref
    return car_data

def process_all_channels(raw_data, sampling_rate, excluded_channels, output_dir='clustering_results'):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    n_channels = raw_data.shape[1]
    included_channels = [ch for ch in range(n_channels) if ch not in excluded_channels]
    
    print(f"\n{'='*60}\nPROCESSING {len(included_channels)} CHANNELS\n{'='*60}")
    
    all_results = {}
    for channel_id in included_channels:
        result = cluster_channel_with_tridesclous(channel_id, raw_data, sampling_rate, output_dir)
        if result: all_results[channel_id] = result
    
    summary = {
        'n_channels_processed': len(all_results),
        'sampling_rate': float(sampling_rate),
        'timestamp': datetime.now().isoformat(),
        'channels': {}
    }
    
    for ch, res in all_results.items():
        summary['channels'][str(ch)] = {
            'n_clusters': res['n_clusters'],
            'cluster_metrics': res['cluster_metrics'],
            'channel_metrics': res['channel_metrics']
        }
    
    with open(os.path.join(output_dir, 'processing_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return all_results

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    data_folder = r"C:\Users\perod\WaveletMatching\mouse46\2014-05-29_10-38-44"
    if not os.path.exists(data_folder):
        data_folder = input("Enter data folder path: ").strip().replace('"', '')

    reader = NeuralynxIO(dirname=data_folder)
    blk = reader.read_block(lazy=False)
    signal = blk.segments[0].analogsignals[0]
    sampling_rate = signal.sampling_rate.rescale('Hz').item()
    raw_data = np.array(signal)

    print(f"Data shape: {raw_data.shape}, Sampling rate: {sampling_rate} Hz")

    excluded_channels = get_excluded_channels(raw_data.shape[1])
    car_groups = get_car_groups(raw_data.shape[1], excluded_channels)
    car_data = apply_car_to_data(raw_data, car_groups)
    process_all_channels(car_data, sampling_rate, excluded_channels)

    print("\n" + "="*60 + "\nPROCESSING COMPLETE\n" + "="*60)