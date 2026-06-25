from logging import root
import re
from neo.io import NeuralynxIO
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

parent_folder = r"C:\Users\HP\Downloads\neuro_files-main\neuro_files-main\mouse46"
output_root = os.path.join(parent_folder, "output")

sliding_window = 100000
downsample_window = 1000
correlation_threshold = 0.9

os.makedirs(output_root, exist_ok=True)


def has_csc_files(folder_path):
    try:
        files = os.listdir(folder_path)
    except Exception:
        return False
    
    return any(re.match(r"^CSC\d+\.ncs$", f, re.IGNORECASE) for f in files)

def extract_channel_number(name):
    match = re.search(r"\d+", str(name))
    if not match:
        raise ValueError(f"Could not extract channel number from name: {name}")
    return int(match.group())

def build_groups(signal_downsample, signal_peaks_mean_sorted_index, threshold=0.9):
    correlation_matrix = np.corrcoef(signal_downsample, rowvar=False)
    print("Correlation matrix done.")

    high_correlation = correlation_matrix > threshold

    non_group_0 = list(signal_peaks_mean_sorted_index)
    final_groups = []

    while len(non_group_0) > 0:
        anchor = non_group_0[0]
        current_group = [anchor]
        remaining = []

        for ch in non_group_0[1:]:
            if high_correlation[anchor, ch]:
                current_group.append(ch)
            else:
                remaining.append(ch)

        final_groups.append(current_group)
        non_group_0 = remaining

    return final_groups, correlation_matrix


def process_dataset(data_folder, output_dir):
    print(f"\nProcessing folder: {data_folder}")

    try:
        reader = NeuralynxIO(dirname=data_folder)
        blk = reader.read_block(lazy=False)

        print("Extracting signals...")
        analogsignals = blk.segments[0].analogsignals

        if len(analogsignals) == 0:
            print(f"Skipping {data_folder}: no analog signals found.")
            return

        signal = analogsignals[0]
        raw_data = np.array(signal)

        names = signal.array_annotations.get('channel_names', None)
        if names is None:
            print(f"Skipping {data_folder}: no channel names found.")
            return

        channel_numbers = [extract_channel_number(name) for name in names]
        order = np.argsort(channel_numbers)
        raw_data = raw_data[:, order]
        channel_names = np.array(names)[order]

        print("Corrected channel order:", channel_names)
        print(f"Raw data shape: {raw_data.shape}")

        raw_data = raw_data * 3.05e-8 * 1e6

        sampling_rate = float(analogsignals[0].sampling_rate)
        time = analogsignals[0].times.rescale('s').magnitude

        print("Data loaded successfully.")
        print(f"Sampling rate: {sampling_rate} Hz")

        signal_rectified = np.abs(raw_data)
        print("Computing moving average (this may take a while)...")
        signal_moving_mean = (
            pd.DataFrame(signal_rectified)
            .rolling(window=sliding_window, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        print("Moving average complete.")

        signal_downsample = signal_moving_mean[::downsample_window]
        time_downsample = time[::downsample_window]
        raw_downsampled = raw_data[::downsample_window]

        plot_channels = list(range(raw_data.shape[1]))

        y_max_raw_downsample = np.max(np.abs(raw_downsampled[:, plot_channels]))

        plt.figure(figsize=(12, 2 * len(plot_channels)))
        plt.suptitle("Downsampled Raw (uV)")

        for i in range(len(plot_channels)):
            plt.subplot(len(plot_channels), 1, i + 1)
            plt.plot(time_downsample, raw_downsampled[:, plot_channels[i]])
            plt.ylabel(f"Ch {plot_channels[i] + 1}")
            plt.ylim(-y_max_raw_downsample, y_max_raw_downsample)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(os.path.join(output_dir, "downsampled_raw.png"), dpi=200)
        plt.close()


        signal_sorted = np.sort(signal_downsample, axis=0)[::-1]
        n = max(1, round(signal_sorted.shape[0] / 100))
        signal_peaks = signal_sorted[:n, :]
        signal_peaks_mean = np.mean(signal_peaks, axis=0)
        signal_peaks_mean_sorted_index = np.argsort(signal_peaks_mean)[::-1]

    
        groups, correlation_matrix = build_groups(
            signal_downsample,
            signal_peaks_mean_sorted_index,
            threshold=correlation_threshold
        )

        results_txt = os.path.join(output_dir, "channel_groups.txt")
        with open(results_txt, "w", encoding="utf-8") as f:
            f.write(f"Dataset: {data_folder}\n")
            f.write(f"Sampling rate: {sampling_rate} Hz\n")
            f.write(f"Channels found: {len(channel_names)}\n\n")
            f.write("Corrected channel order:\n")
            f.write(", ".join(map(str, channel_names)) + "\n\n")

            f.write("Final channel groupings:\n")
            for i, group in enumerate(groups, start=1):
                group_1based = np.array(group) + 1
                f.write(f"Group {i}: {group_1based}\n")

       
        corr_df = pd.DataFrame(
            correlation_matrix,
            index=[f"Ch{i+1}" for i in range(correlation_matrix.shape[0])],
            columns=[f"Ch{i+1}" for i in range(correlation_matrix.shape[1])]
        )
        corr_df.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))


        peak_df = pd.DataFrame({
            "Channel": [f"Ch{i+1}" for i in range(len(signal_peaks_mean))],
            "PeakMean": signal_peaks_mean
        }).sort_values("PeakMean", ascending=False)
        peak_df.to_csv(os.path.join(output_dir, "peak_means.csv"), index=False)

        print("Final channel groupings:")
        for i, group in enumerate(groups, start=1):
            print(f"Group {i}: {np.array(group) + 1}")

        print(f"Finished processing: {data_folder}")
        print(f"Saved outputs to: {output_dir}")

    except Exception as e:
        print(f"Error processing {data_folder}: {e}")


def process_all_subfolders(parent_folder, output_root):
    print(f"Scanning parent folder:\n{parent_folder}")

    found_any = False
    parent_folder_abs = os.path.abspath(parent_folder)
    output_root_abs = os.path.abspath(output_root)

    for root, dirs, files in os.walk(parent_folder_abs):
       root_abs = os.path.abspath(root)
       
       try:
            if os.path.commonpath([root_abs, output_root_abs]) == output_root_abs:
                continue
       except ValueError:
            pass

       if has_csc_files(root):
            found_any = True
            relative_path = os.path.relpath(root, parent_folder_abs)
            safe_name = relative_path.replace("\\", "_").replace("/", "_")
            dataset_output_dir = os.path.join(output_root_abs, safe_name)
            os.makedirs(dataset_output_dir, exist_ok=True)

            process_dataset(root, dataset_output_dir)

    if not found_any:
        print("No subfolders containing CSC#.ncs files were found.")



if __name__ == "__main__":
    process_all_subfolders(parent_folder, output_root)