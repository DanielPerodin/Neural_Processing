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


def get_preprocessor_params(fs, n_channels):
    """
    Get the standard Tridesclous preprocessor params for the given
    recording without creating any processed data on disk.
    We only need the 'preprocessor' sub-dict.
    """
    import shutil
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
    finally:
        try:
            shutil.rmtree(temp_folder)
        except:
            pass


# ==========================================
# METRIC CALCULATION
# ==========================================

def calculate_metrics(ch_data_snippet, fs, preprocessor_params):
    """
    Compute noise and SNR metrics for a single channel snippet.

    Pipeline
    --------
    Uses two Tridesclous internal functions directly on the numpy array —
    no DataIO, no temp folders, no CatalogueConstructor:

      offline_signal_preprocessor(..., normalize=False)
          Applies the same bandpass filter (300-5000 Hz, 5th-order Butterworth,
          filtfilt) that Tridesclous uses internally, but returns values in
          the original µV units rather than normalizing.

      estimate_medians_mads_after_preprocesing(...)
          Filters with normalize=False and returns (median, MAD) per channel,
          which is Tridesclous's own noise estimate.

    Metrics
    -------
    MAD (µV)
        Tridesclous's native noise unit. median(|x - median(x)|) of the
        filtered signal, computed by Tridesclous's own median_mad function.

    Threshold
        -3 × MAD (downward deflections).
        Per Tridesclous docs: 3 MAD = 99.7% of Gaussian noise.

    Noise samples  : filtered_sig >= threshold  (baseline, 99.7%)
    Spike samples  : filtered_sig <  threshold  (putative spikes)

    RMS noise (µV)  [PI's metric]
        RMS of noise samples only — baseline power after spike exclusion.

    RMS signal (µV)
        RMS of spike samples only.

    SNR
        RMS signal / RMS noise.

    Args
    ----
    ch_data_snippet   : np.ndarray, shape (n_samples, 1), float32, µV
    fs                : float, Hz
    preprocessor_params : dict from tdc.get_auto_params_for_catalogue

    Returns
    -------
    dict with keys:
        mad_uV, rms_noise_uV, rms_signal_uV, snr, spike_sample_count
    """
    results = {
        'mad_uV':             np.nan,
        'rms_noise_uV':       np.nan,
        'rms_signal_uV':      np.nan,
        'snr':                np.nan,
        'spike_sample_count': 0,
    }

    try:
        # --- Filter only, keep µV units ---
        filtered = offline_signal_preprocessor(
            ch_data_snippet,
            fs,
            normalize=False,
            **preprocessor_params
        )
        flat = filtered[:, 0]

        # --- MAD: Tridesclous's own noise estimate ---
        # estimate_medians_mads_after_preprocesing filters then calls
        # median_mad(), which computes median(|x - median(x)|) per channel
        med, mad = estimate_medians_mads_after_preprocesing(
            ch_data_snippet,
            fs,
            **preprocessor_params
        )
        mad_uv = float(mad[0])
        results['mad_uV'] = mad_uv

        # --- Split into noise and spike samples at 3×MAD ---
        threshold = -3.0 * mad_uv
        noise_samples = flat[flat >= threshold]
        spike_samples = flat[flat <  threshold]
        results['spike_sample_count'] = int(spike_samples.shape[0])

        # --- RMS noise: PI's metric ---
        if noise_samples.shape[0] > 0:
            results['rms_noise_uV'] = float(
                np.sqrt(np.mean(noise_samples ** 2))
            )

        # --- RMS signal and SNR ---
        if spike_samples.shape[0] > 0:
            rms_signal = float(np.sqrt(np.mean(spike_samples ** 2)))
            results['rms_signal_uV'] = rms_signal
            if results['rms_noise_uV'] > 0:
                results['snr'] = rms_signal / results['rms_noise_uV']
        else:
            results['rms_signal_uV'] = 0.0
            results['snr'] = 0.0

    except Exception as e:
        print(f"\n  [metric error] {e}")
        traceback.print_exc()

    return results


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("NOISE & SNR CALCULATOR")
    print(f"Snippet : {SNIPPET_DURATION_S}s from middle of recording")
    print("Filter  : Tridesclous offline_signal_preprocessor (normalize=False)")
    print("MAD     : Tridesclous estimate_medians_mads_after_preprocesing")
    print("Noise   : RMS of samples within 3×MAD (baseline only)")
    print("Signal  : RMS of samples beyond 3×MAD (spikes only)")
    print("SNR     : RMS signal / RMS noise")
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

    # 3. Get Tridesclous preprocessor params once (shared across all folders/channels)
    print("\nFetching Tridesclous preprocessor params...")
    preprocessor_params = get_preprocessor_params(fs_global, n_channels)
    print(f"  {preprocessor_params}")

    # 4. User configuration (done once, applied to all folders)
    excluded_channels = get_excluded_channels(n_channels)
    car_groups        = get_car_groups(n_channels, excluded_channels)
    included_channels = [ch for ch in range(n_channels) if ch not in excluded_channels]

    print(f"\nStarting analysis ({SNIPPET_DURATION_S}s middle snippet per recording)...")
    print("-" * 60)

    all_rows = []

    # 5. Main processing loop
    for folder in subfolders:
        folder_name = os.path.basename(folder)

        if not any(f.endswith('.ncs') for f in os.listdir(folder)):
            continue

        print(f"\nProcessing: {folder_name}")

        try:
            # A. Load middle snippet only
            print(f"  > Loading {SNIPPET_DURATION_S}s from middle...", end="\r")
            raw_snippet, fs = load_middle_snippet(folder, SNIPPET_DURATION_S)
            n_loaded_s = raw_snippet.shape[0] / fs
            print(f"  > Loaded {n_loaded_s:.1f}s  "
                  f"({raw_snippet.shape[0]} samples @ {fs:.0f} Hz)")

            # B. Apply CAR across all included channels
            car_snippet = apply_car_to_data(raw_snippet, car_groups)

            row = {'Date_Folder': folder_name}

            # C. Per-channel metrics
            for ch in included_channels:
                print(f"  > Analyzing Channel {ch}...          ", end="\r")

                # shape (n_samples, 1) — offline_signal_preprocessor expects 2D
                ch_data = car_snippet[:, ch:ch+1]
                m = calculate_metrics(ch_data, fs, preprocessor_params)

                p = f"Ch{ch}"
                row[f'{p}_MAD_uV']        = round(m['mad_uV'],        3)
                row[f'{p}_RMS_Noise_uV']  = round(m['rms_noise_uV'],  3)
                row[f'{p}_RMS_Signal_uV'] = round(m['rms_signal_uV'], 3)
                row[f'{p}_SNR']           = round(m['snr'],           3)
                row[f'{p}_SpikeSamples']  = m['spike_sample_count']

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