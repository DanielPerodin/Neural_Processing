from logging import root
import re
from neo.io import NeuralynxIO
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import json

parent_folder = r"E:\FDA Raw Data\ephys\mouse36"
output_root = os.path.join(parent_folder, "output")
print(parent_folder)
print(os.path.exists(parent_folder))
print(os.listdir(parent_folder))

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
        blk = reader.read_block(lazy=True)

        analogsignals = blk.segments[0].analogsignals

        if len(analogsignals) == 0:
            print("No analog signals.")
            return

        signal = analogsignals[0]

        names = signal.array_annotations.get("channel_names", None)
        if names is None:
            print("No channel names.")
            return

        channel_numbers = [extract_channel_number(name) for name in names]
        order = np.argsort(channel_numbers)
        channel_names = np.array(names)[order]

        sampling_rate = float(signal.sampling_rate)

        print(f"Sampling rate: {sampling_rate}")

        chunk_seconds = 30

        start = signal.t_start
        stop = signal.t_stop

        downsampled_chunks = []
        raw_plot_chunks = []
        time_chunks = []

        while start < stop:

            end = min(start + chunk_seconds * signal.t_start.units, stop)

            print(f"Loading {start} -> {end}")

            chunk = signal.load(time_slice=(start, end))

            raw = np.asarray(chunk)

            raw = raw[:, order]

            raw = raw * 3.05e-8 * 1e6

            raw_plot_chunks.append(raw[::downsample_window])

            time_chunks.append(
                chunk.times.rescale("s").magnitude[::downsample_window]
            )

            rectified = np.abs(raw).astype(np.float32)

            moving = np.empty_like(rectified, dtype=np.float32)

            for ch in range(rectified.shape[1]):
                moving[:, ch] = uniform_filter1d(
                    rectified[:, ch],
                    size=sliding_window,
                    mode="nearest"
                )

            analysis_downsample = 1000   # used for CAR grouping
            plot_downsample = 5000       # used only for plotting

            downsampled_chunks.append(
                moving[::downsample_window]
            )

            raw_plot_chunks.append(
                raw[::plot_downsample]
            )

            time_chunks.append(
                chunk.times.rescale("s").magnitude[::plot_downsample]
            )

            del raw
            del rectified
            del moving
            del chunk

            start = end

        signal_downsample = np.vstack(downsampled_chunks)
        raw_downsampled = np.vstack(raw_plot_chunks)
        time_downsample = np.concatenate(time_chunks)

        print("Finished loading all chunks.")

        plot_channels = list(range(raw_downsampled.shape[1]))

        y_max = np.max(np.abs(raw_downsampled))

        plt.figure(figsize=(12, 2 * len(plot_channels)))
        plt.suptitle("Downsampled Raw (uV)")

        for i in plot_channels:
            plt.subplot(len(plot_channels), 1, i + 1)
            plt.plot(time_downsample, raw_downsampled[:, i])
            plt.ylabel(f"Ch {i+1}")
            plt.ylim(-y_max, y_max)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "downsampled_raw.png"))
        plt.close()

        signal_sorted = np.sort(signal_downsample, axis=0)[::-1]
        n = max(1, round(signal_sorted.shape[0] / 100))
        signal_peaks = signal_sorted[:n]
        signal_peaks_mean = np.mean(signal_peaks, axis=0)
        signal_peaks_mean_sorted_index = np.argsort(signal_peaks_mean)[::-1]

        groups, correlation_matrix = build_groups(
            signal_downsample,
            signal_peaks_mean_sorted_index,
            threshold=correlation_threshold,
        )

        print(groups)

        groups_to_save = [[int(ch) for ch in group] for group in groups]

        with open(os.path.join(output_dir, "car_groups.json"), "w") as f:
            json.dump(groups_to_save, f, indent=2)

        print("Saved CAR groups.")

    except Exception as e:
        print(e)


def process_all_subfolders(parent_folder, output_root):
    print(f"Scanning parent folder:\n{parent_folder}")

    found_any = False
    parent_folder_abs = os.path.abspath(parent_folder)
    output_root_abs = os.path.abspath(output_root)

    for root, dirs, files in os.walk(parent_folder_abs):
        print("Checking:", root)
        print(files)

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