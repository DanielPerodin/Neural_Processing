import os
import numpy as np
import pandas as pd
import traceback
import quantities as pq
from neo.io import NeuralynxIO
from tridesclous.signalpreprocessor import (
    offline_signal_preprocessor,
    estimate_medians_mads_after_preprocesing,
)
import tridesclous as tdc
import tempfile
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
SNIPPET_DURATION_S = 60  # seconds sampled from the middle of each recording

# ==========================================
# CHANNEL / CAR CONFIGURATION (from clustering6.py)
# ==========================================

def get_excluded_channels(n_channels):
    """Prompt user to specify channels to exclude from processing"""
    print(f"\n{'='*60}")
    print(f"CHANNEL EXCLUSION")
    print(f"{'='*60}")
    print(f"Total channels available: 0 to {n_channels-1}")
    print("\nWould you like to EXCLUDE any channels from processing?")
    print("(Excluded channels will not be processed and will not affect CAR)")

    response = input("\nExclude channels? (y/n): ").lower().strip()

    if response not in ['y', 'yes']:
        print("No channels will be excluded.")
        return []

    print("\nEnter channel IDs to exclude (comma-separated, e.g., '0,3,7'):")
    excluded_input = input("Excluded channels: ").strip()

    if not excluded_input:
        print("No channels excluded.")
        return []

    try:
        excluded = [int(ch.strip()) for ch in excluded_input.split(',')]
        excluded = [ch for ch in excluded if 0 <= ch < n_channels]
        print(f"Excluded channels: {sorted(excluded)}")
        return excluded
    except:
        print("Invalid input. No channels will be excluded.")
        return []


def get_car_groups(n_channels, excluded_channels):
    """Prompt user to define CAR groupings"""
    included_channels = [ch for ch in range(n_channels) if ch not in excluded_channels]

    print(f"\n{'='*60}")
    print(f"COMMON AVERAGE REFERENCE (CAR) GROUPING")
    print(f"{'='*60}")
    print(f"Channels to process: {included_channels}")
    print(f"\nWould you like to group channels for CAR computation?")
    print("Options:")
    print("  1. Apply CAR using ALL included channels (default)")
    print("  2. Create custom CAR groups (channels within a group share CAR)")
    print("  3. No CAR grouping (each channel gets individual CAR)")

    response = input("\nEnter choice (1/2/3): ").strip()

    if response == '2':
        print("\nDefine CAR groups:")
        print("Enter groups as comma-separated channel lists, one group per line.")
        print("Example:")
        print("  0,1,2,3")
        print("  4,5,6,7")
        print("(Press Enter on empty line when done)")

        groups = []
        while True:
            group_input = input(f"Group {len(groups)+1}: ").strip()
            if not group_input:
                break
            try:
                group = [int(ch.strip()) for ch in group_input.split(',')]
                group = [ch for ch in group if ch in included_channels]
                if group:
                    groups.append(group)
                    print(f"  Added group {len(groups)}: {group}")
            except:
                print("  Invalid input, skipping this group")

        if groups:
            grouped_channels = set([ch for group in groups for ch in group])
            ungrouped = [ch for ch in included_channels if ch not in grouped_channels]

            if ungrouped:
                print(f"\nWarning: Channels {ungrouped} are not in any group.")
                print("These channels will get individual CAR.")
                for ch in ungrouped:
                    groups.append([ch])

            print(f"\nFinal CAR groups: {groups}")
            return groups
        else:
            print("No valid groups created. Using default (all channels).")
            return [included_channels]

    elif response == '3':
        print("\nNo CAR grouping. Each channel will get individual CAR.")
        return [[ch] for ch in included_channels]

    else:
        print(f"\nUsing default: CAR from all {len(included_channels)} included channels.")
        return [included_channels]


def apply_car_to_data(raw_data, car_groups):
    """Apply common average reference based on defined groups"""
    car_data = raw_data.copy()

    for group in car_groups:
        if len(group) == 1:
            ch = group[0]
            car_data[:, ch] -= raw_data[:, ch].mean()
        else:
            car_reference = raw_data[:, group].mean(axis=1, keepdims=True)
            car_data[:, group] = raw_data[:, group] - car_reference

    return car_data


# ==========================================
# DATA LOADING
# ==========================================

def load_middle_snippet(folder, snippet_duration_s):
    """
    Load only a central time window from the recording without pulling
    the entire file into RAM.

    Neo's AnalogSignalProxy.load(time_slice=(t_start, t_stop)) reads only
    the requested window from disk when given Quantity time values.

    Returns
    -------
    raw_uV : np.ndarray, shape (n_samples, n_channels), float32
    fs     : float, sampling rate in Hz
    """
    reader = NeuralynxIO(dirname=folder)
    blk = reader.read_block(lazy=True)
    proxy = blk.segments[0].analogsignals[0]

    fs = float(proxy.sampling_rate.rescale('Hz'))
    total_samples = proxy.shape[0]
    total_duration_s = total_samples / fs

    actual_duration_s = min(snippet_duration_s, total_duration_s)
    half_s = actual_duration_s / 2.0
    mid_s  = total_duration_s  / 2.0

    t_start = max(0.0, mid_s - half_s) * pq.s
    t_stop  = min(total_duration_s, mid_s + half_s) * pq.s

    snippet = proxy.load(time_slice=(t_start, t_stop))
    raw_uV  = snippet.rescale('uV').magnitude.astype('float32')

    return raw_uV, fs


def load_full_recording(folder):
    """
    Load the entire recording into memory for spike detection.
    Returns raw_uV (n_samples, n_channels) float32 and fs.
    """
    reader = NeuralynxIO(dirname=folder)
    blk = reader.read_block(lazy=False)
    signal = blk.segments[0].analogsignals[0]
    fs = float(signal.sampling_rate.rescale('Hz'))
    raw_uV = np.array(signal.rescale('uV')).astype('float32')
    return raw_uV, fs


# ==========================================
# METRIC CALCULATION
# ==========================================


# ==========================================
# TRIDESCLOUS HELPERS
# ==========================================

def get_preprocessor_params(fs, n_channels=1):
    """
    Fetch Tridesclous auto preprocessor params for the given sampling rate.
    Creates a throw-away DataIO with a dummy file, reads the params, deletes everything.
    """
    temp_folder = tempfile.mkdtemp(prefix="tdc_params_")
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
        try:
            shutil.rmtree(temp_folder)
        except Exception:
            pass


def calculate_channel_noise_metrics(channel_data, fs, preprocessor_params):
    """
    Compute MAD and RMS noise floor for a single-channel snippet.
    Signal and SNR are left as 0.0 — they are filled in after spike detection
    by compute_snr_from_waveforms(), so SNR is always grounded in actual
    Tridesclous-detected spikes rather than a threshold heuristic.

    Returns dict: {mad_uV, rms_noise_uV, mean_peak_signal_uV=0, channel_snr=0}
    """
    results = {
        'mad_uV':              0.0,
        'rms_noise_uV':        1.0,
        'mean_peak_signal_uV': 0.0,
        'channel_snr':         0.0,
    }
    try:
        data_2d = channel_data[:, None] if channel_data.ndim == 1 else channel_data

        filtered = offline_signal_preprocessor(
            data_2d, fs, normalize=False, **preprocessor_params
        )
        flat = filtered[:, 0]

        _, mad = estimate_medians_mads_after_preprocesing(
            data_2d, fs, **preprocessor_params
        )
        mad_uv = float(mad[0])
        results['mad_uV'] = mad_uv

        noise_samples = flat[flat >= -3.0 * mad_uv]
        if noise_samples.shape[0] > 0:
            results['rms_noise_uV'] = float(np.sqrt(np.mean(noise_samples ** 2)))
    except Exception as e:
        print(f"  [noise metric error] {e}")
        traceback.print_exc()
    return results


def compute_snr_from_waveforms(waveforms, channel_metrics):
    """
    Fill mean_peak_signal_uV and channel_snr into channel_metrics (in-place)
    using the trough of each detected spike waveform as the signal measure.
        signal = |mean( min(waveform) for each spike )|
        SNR    = signal / rms_noise_uV
    Returns 0 for both if no waveforms are provided.
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


def detect_spikes_and_waveforms(channel_data, sampling_rate):
    """
    Run the Tridesclous detection + peeling pipeline on a single channel and
    return extracted waveforms — no plotting, no JSON, no cluster output.
    Used to compute SNR from actual detected spikes.

    Returns dict:
        waveforms   – np.ndarray (n_spikes, n_template_samples), or empty
        spike_times – np.ndarray (n_spikes,), seconds
        n_spikes    – int
        status      – 'ok' | 'no_spikes' | 'error'
    """
    empty = {'waveforms': np.empty((0,), dtype='float32'),
             'spike_times': np.empty((0,), dtype='float64'),
             'n_spikes': 0, 'status': 'no_spikes'}

    if channel_data.ndim == 1:
        channel_data = channel_data[:, None]
    channel_data = channel_data.astype('float32')

    temp_folder = tempfile.mkdtemp(prefix="tdc_detect_")
    try:
        raw_file = os.path.join(temp_folder, 'raw.raw')
        channel_data.tofile(raw_file)

        dataio = tdc.DataIO(dirname=temp_folder)
        dataio.set_data_source(type='RawData', filenames=[raw_file],
                               dtype='float32', sample_rate=sampling_rate,
                               total_channel=1)
        dataio.add_one_channel_group(channels=[0])

        cc = tdc.CatalogueConstructor(dataio=dataio, chan_grp=0)
        params = tdc.get_auto_params_for_catalogue(dataio, chan_grp=0)
        cc.apply_all_steps(params, verbose=False)
        cc.make_catalogue_for_peeler()

        catalogue = dataio.load_catalogue(chan_grp=0)
        peeler = tdc.Peeler(dataio)
        peeler.change_params(catalogue=catalogue)
        peeler.run(progressbar=False)

        spikes = dataio.get_spikes(seg_num=0, chan_grp=0).copy()
        if len(spikes) == 0:
            return empty

        n_template_samples = catalogue['centers0'].shape[1]
        pre_samples  = n_template_samples // 2
        post_samples = n_template_samples - pre_samples
        n_total      = len(channel_data)

        waveforms, spike_times = [], []
        for spike in spikes:
            idx   = spike['index']
            label = spike['cluster_label']
            s, e  = idx - pre_samples, idx + post_samples
            if s >= 0 and e <= n_total and label >= 0:
                waveforms.append(channel_data[s:e, 0])
                spike_times.append(idx / sampling_rate)

        if not waveforms:
            return empty

        return {'waveforms':   np.array(waveforms,   dtype='float32'),
                'spike_times': np.array(spike_times, dtype='float64'),
                'n_spikes':    len(waveforms),
                'status':      'ok'}
    except Exception as e:
        print(f"  [spike detection error] {e}")
        traceback.print_exc()
        return {**empty, 'status': 'error'}
    finally:
        try:
            shutil.rmtree(temp_folder)
        except Exception:
            pass

# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("NOISE & SNR CALCULATOR")
    print(f"Noise   : {SNIPPET_DURATION_S}s middle snippet → MAD + RMS noise floor")
    print("Spikes  : full recording → Tridesclous detection + peeling")
    print("Signal  : |mean of per-spike trough amplitudes| across full recording")
    print("SNR     : |mean spike trough| / RMS noise  (0 if no spikes detected)")
    print("=" * 60)

    # 1. Path input
    parent_dir = input("\nEnter Mouse Folder Path (e.g. .../mouse46): ").strip().replace('"', '')
    if not os.path.exists(parent_dir):
        print("Folder not found.")
        return

    subfolders = sorted([f.path for f in os.scandir(parent_dir) if f.is_dir()])
    if not subfolders:
        print("No subfolders found.")
        return

    # 2. Detect channel count and sampling rate from first valid folder
    try:
        n_channels = None
        fs_global  = None
        for folder in subfolders:
            if any(f.endswith('.ncs') for f in os.listdir(folder)):
                reader = NeuralynxIO(dirname=folder)
                blk = reader.read_block(lazy=True)
                proxy = blk.segments[0].analogsignals[0]
                n_channels = proxy.shape[1]
                fs_global  = float(proxy.sampling_rate.rescale('Hz'))
                print(f"\nDetected {n_channels} channels @ {fs_global:.0f} Hz.")
                break

        if n_channels is None:
            print("No folders with .ncs files found.")
            return

    except Exception as e:
        print(f"Error reading metadata: {e}")
        traceback.print_exc()
        return

    # 3. Get Tridesclous preprocessor params once (used for noise metric filtering)
    print("\nFetching Tridesclous preprocessor params...")
    preprocessor_params = get_preprocessor_params(fs_global, n_channels)
    print(f"  {preprocessor_params}")

    print(f"\nStarting analysis ({SNIPPET_DURATION_S}s middle snippet per recording)...")
    print("-" * 60)

    all_rows = []

    # 4. Main processing loop
    for folder in subfolders:
        folder_name = os.path.basename(folder)

        if not any(f.endswith('.ncs') for f in os.listdir(folder)):
            continue

        print(f"\nProcessing: {folder_name}")

        # 5. Per-folder CAR configuration
        excluded_channels = get_excluded_channels(n_channels)
        car_groups        = get_car_groups(n_channels, excluded_channels)
        included_channels = [ch for ch in range(n_channels) if ch not in excluded_channels]

        try:
            # A. Load 60s snippet for noise metrics
            print(f"  > Loading {SNIPPET_DURATION_S}s snippet (noise metrics)...", end="\r")
            raw_snippet, fs = load_middle_snippet(folder, SNIPPET_DURATION_S)
            n_loaded_s = raw_snippet.shape[0] / fs
            print(f"  > Snippet: {n_loaded_s:.1f}s  "
                  f"({raw_snippet.shape[0]} samples @ {fs:.0f} Hz)")

            # B. Load full recording for spike detection
            print(f"  > Loading full recording (spike detection)...", end="\r")
            raw_full, _ = load_full_recording(folder)
            print(f"  > Full recording: {raw_full.shape[0] / fs:.1f}s  "
                  f"({raw_full.shape[0]} samples)")

            # C. Apply CAR to both
            car_snippet = apply_car_to_data(raw_snippet, car_groups)
            car_full    = apply_car_to_data(raw_full,    car_groups)

            row = {'Date_Folder': folder_name}

            # D. Per-channel metrics
            for ch in included_channels:
                print(f"  > Analyzing Channel {ch}...          ", end="\r")

                ch_snippet = car_snippet[:, ch:ch+1]  # 60s — noise metrics only
                ch_full    = car_full[:,    ch:ch+1]  # full recording — spike detection

                # Noise metrics (MAD + RMS noise floor) from 60s snippet
                noise_m = calculate_channel_noise_metrics(
                    ch_snippet, fs, preprocessor_params
                )

                # Spike detection from full recording → SNR grounded in all spikes
                spike_result = detect_spikes_and_waveforms(ch_full, fs)
                compute_snr_from_waveforms(spike_result['waveforms'], noise_m)

                p = f"Ch{ch}"
                row[f'{p}_MAD_uV']       = round(noise_m['mad_uV'],               3)
                row[f'{p}_RMS_Noise_uV'] = round(noise_m['rms_noise_uV'],         3)
                row[f'{p}_MeanPeak_uV']  = round(noise_m['mean_peak_signal_uV'],  3)
                row[f'{p}_SNR']          = round(noise_m['channel_snr'],          3)
                row[f'{p}_N_Spikes']     = spike_result['n_spikes']
                row[f'{p}_DetectStatus'] = spike_result['status']

            print(f"  > Done.                                    ")
            all_rows.append(row)

        except Exception as e:
            print(f"  > Failed: {e}")
            traceback.print_exc()

    # 6. Save results
    if all_rows:
        df = pd.DataFrame(all_rows)
        out_file = os.path.join(parent_dir, 'noise_snr_analysis_60s.csv')
        df.to_csv(out_file, index=False)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Saved: {out_file}")
        print("\nFirst 5 rows:")
        print(df.head())
    else:
        print("No valid data processed.")


if __name__ == "__main__":
    main()