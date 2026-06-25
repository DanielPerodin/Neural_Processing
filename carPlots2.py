import os
import numpy as np
import quantities as pq
from neo.io import NeuralynxIO

# ==========================================
# CONFIGURATION
# ==========================================
SNIPPET_DURATION_S = 2.0   # seconds to plot (taken from the middle of each recording)
PLOT_DPI           = 150

# ==========================================
# CHANNEL / CAR CONFIGURATION
# ==========================================

def get_excluded_channels(n_channels):
    """Prompt user to specify channels to exclude from processing."""
    print(f"\n{'='*60}")
    print("CHANNEL EXCLUSION")
    print(f"{'='*60}")
    print(f"Total channels available: 0 to {n_channels - 1}")
    print("\nWould you like to EXCLUDE any channels from processing?")

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
    except Exception:
        print("Invalid input. No channels will be excluded.")
        return []


def get_car_groups(n_channels, excluded_channels):
    """Prompt user to define CAR groupings."""
    included_channels = [ch for ch in range(n_channels) if ch not in excluded_channels]

    print(f"\n{'='*60}")
    print("COMMON AVERAGE REFERENCE (CAR) GROUPING")
    print(f"{'='*60}")
    print(f"Channels to process: {included_channels}")
    print("\nWould you like to group channels for CAR computation?")
    print("  1. Apply CAR using ALL included channels (default)")
    print("  2. Create custom CAR groups")
    print("  3. No CAR grouping (each channel gets individual CAR)")

    response = input("\nEnter choice (1/2/3): ").strip()

    if response == '2':
        print("\nDefine CAR groups (comma-separated, one group per line; blank line to finish):")
        groups = []
        while True:
            group_input = input(f"Group {len(groups) + 1}: ").strip()
            if not group_input:
                break
            try:
                group = [int(ch.strip()) for ch in group_input.split(',')]
                group = [ch for ch in group if ch in included_channels]
                if group:
                    groups.append(group)
                    print(f"  Added group {len(groups)}: {group}")
            except Exception:
                print("  Invalid input, skipping this group.")

        if groups:
            grouped_channels = {ch for group in groups for ch in group}
            ungrouped = [ch for ch in included_channels if ch not in grouped_channels]
            if ungrouped:
                print(f"\nWarning: channels {ungrouped} are not in any group — assigning individual CAR.")
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
    """Apply common average reference based on defined groups."""
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
    the entire file into RAM (same approach as noiseFloor.py).

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


# ==========================================
# PLOTTING
# ==========================================

def save_car_comparison_plots(raw_snippet, car_snippet, fs, included_channels,
                               folder_name, output_dir):
    """
    Save one PNG per included channel with pre-CAR (blue) and post-CAR
    (orange) traces stacked vertically and sharing an x-axis.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_samples = raw_snippet.shape[0]
    t_ms = (np.arange(n_samples) / fs) * 1000.0  # ms

    for ch in included_channels:
        raw_trace = raw_snippet[:, ch]
        car_trace = car_snippet[:, ch]

        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        fig.suptitle(
            f"{folder_name}  |  Channel {ch} — Pre vs Post CAR"
            f"  ({n_samples / fs:.1f}s snippet from middle)",
            fontsize=12, fontweight='bold'
        )

        axes[0].plot(t_ms, raw_trace, color='steelblue', linewidth=0.5)
        axes[0].set_ylabel('Amplitude (µV)')
        axes[0].set_title('Pre-CAR (raw)')
        axes[0].grid(True, alpha=0.3, linestyle='--')

        axes[1].plot(t_ms, car_trace, color='darkorange', linewidth=0.5)
        axes[1].set_ylabel('Amplitude (µV)')
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_title('Post-CAR')
        axes[1].grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        fname = os.path.join(output_dir, f'channel_{ch}_pre_post_CAR.png')
        plt.savefig(fname, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {fname}")


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("PRE / POST-CAR PLOT GENERATOR")
    print(f"Snippet : {SNIPPET_DURATION_S}s from middle of each recording")
    print("Output  : one PNG per channel per session folder")
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

    # 2. Detect channel count from first valid folder
    n_channels = None
    fs_global  = None
    for folder in subfolders:
        if any(f.endswith('.ncs') for f in os.listdir(folder)):
            try:
                reader = NeuralynxIO(dirname=folder)
                blk = reader.read_block(lazy=True)
                proxy = blk.segments[0].analogsignals[0]
                n_channels = proxy.shape[1]
                fs_global  = float(proxy.sampling_rate.rescale('Hz'))
                print(f"\nDetected {n_channels} channels @ {fs_global:.0f} Hz.")
                break
            except Exception as e:
                print(f"Could not read {folder}: {e}")

    if n_channels is None:
        print("No folders with .ncs files found.")
        return

    print(f"\nStarting plot generation ({SNIPPET_DURATION_S}s middle snippet per recording)...")
    print("-" * 60)

    # 3. Main loop — one subfolder = one recording session / day
    for folder in subfolders:
        folder_name = os.path.basename(folder)

        if not any(f.endswith('.ncs') for f in os.listdir(folder)):
            continue

        print(f"\nProcessing: {folder_name}")

        # Per-folder CAR configuration
        excluded_channels = get_excluded_channels(n_channels)
        car_groups        = get_car_groups(n_channels, excluded_channels)
        included_channels = [ch for ch in range(n_channels) if ch not in excluded_channels]

        try:
            # Load middle snippet
            print(f"  > Loading {SNIPPET_DURATION_S}s from middle...", end='\r')
            raw_snippet, fs = load_middle_snippet(folder, SNIPPET_DURATION_S)
            print(f"  > Loaded {raw_snippet.shape[0] / fs:.2f}s  "
                  f"({raw_snippet.shape[0]} samples @ {fs:.0f} Hz)")

            # Apply CAR
            car_snippet = apply_car_to_data(raw_snippet, car_groups)

            # Output goes into <mouse_folder>/<session_name>_CAR_plots/
            output_dir = os.path.join(parent_dir, f"{folder_name}_CAR_plots")
            os.makedirs(output_dir, exist_ok=True)

            # Save one PNG per channel
            save_car_comparison_plots(
                raw_snippet, car_snippet, fs,
                included_channels, folder_name, output_dir
            )

        except Exception as e:
            import traceback
            print(f"  > Failed: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
