import re
from neo.io import NeuralynxIO
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

data_folder = r"data"

print("Loading Neuralynx data...")
reader = NeuralynxIO(dirname=data_folder)
blk = reader.read_block(lazy=False)

print("Extracting signals...")
analogsignals = blk.segments[0].analogsignals

signal = analogsignals[0]
raw_data = np.array(signal)

names = signal.array_annotations['channel_names']
order = np.argsort([int(name.replace('CSC','')) for name in names])

raw_data = raw_data[:, order]
channel_names = names[order]

print("Corrected channel order:", channel_names)

print(f"Raw data shape: {raw_data.shape}")

raw_data = raw_data * 3.05e-8 * 1e6

sampling_rate = float(analogsignals[0].sampling_rate)
time = analogsignals[0].times.rescale('s').magnitude

print("Data loaded successfully.")

signal_rectified = np.abs(raw_data)
sliding_window = 100000
print("Computing moving average (this may take a while)...")
signal_moving_mean = (pd.DataFrame(signal_rectified).rolling(window=sliding_window, center=True, min_periods=1).mean().to_numpy())
print("Moving average complete.")
downsample_window = 1000
signal_downsample = signal_moving_mean[::downsample_window]
time_downsample = time[::downsample_window]
raw_downsampled = raw_data[::downsample_window]
rectified_downsampled = signal_rectified[::downsample_window]

plot_channels = list(range(raw_data.shape[1]))
signal_diff = np.zeros((len(signal_downsample), len(plot_channels), len(plot_channels)))

y_max_raw_downsample = np.max(raw_downsampled[:, plot_channels])

plt.figure()
plt.title("Downsampled Raw (uV)")

for i in range(len(plot_channels)):
    plt.subplot(len(plot_channels), 1, i+1)
    plt.plot(time_downsample, raw_downsampled[:, plot_channels[i]])
    plt.ylabel(f"Ch {plot_channels[i]}")
    plt.ylim(-0.1*y_max_raw_downsample, y_max_raw_downsample)

plt.savefig("downsampled_raw.png", dpi=200)
plt.close()

y_max_rectified_downsampled = np.max(rectified_downsampled[:,plot_channels])
plt.figure()
plt.title("Downsampled Rectified (uV)")

for i in range(len(plot_channels)):
    plt.subplot(len(plot_channels), 1, i+1)
    plt.plot(time_downsample, rectified_downsampled[:, plot_channels[i]])
    plt.ylabel(f"Ch {plot_channels[i]}")
    plt.ylim(-0.1*y_max_raw_downsample, y_max_raw_downsample)

plt.savefig("downsampled_rectified.png", dpi=200)
plt.close()

rectified_downsampled_denoised = rectified_downsampled.copy()
threshold = 3 * np.mean(rectified_downsampled)
rectified_downsampled_denoised[rectified_downsampled_denoised < threshold] = 0

plt.figure()
plt.title("Downsampled Raw (uV)")

for i in range(len(plot_channels)):
    plt.subplot(len(plot_channels), 1, i+1)
    plt.plot(time_downsample, rectified_downsampled_denoised[:, plot_channels[i]])
    plt.ylabel(f"Ch {plot_channels[i]}")
    plt.ylim(-0.1*y_max_raw_downsample, y_max_raw_downsample)

plt.savefig("downsampled_raw2.png", dpi=200)
plt.close()


""" Compute pairwise channel differences
for plot_index in range(len(plot_channels)):
    for plot_diff_index in range(len(plot_channels)):
        signal_diff[:, plot_index, plot_diff_index] = np.abs(
            signal_downsample[:, plot_index] -
            signal_downsample[:, plot_diff_index]
        )

y_max_downsample = np.max(signal_downsample[:, plot_channels])
y_max_diff = np.max(signal_diff[:, plot_channels][:, :, plot_channels]) """

""" Plot rectified downsampled signals
plt.figure()
plt.title("Downsampled Rectified (uV)")
for plot_index in range(len(plot_channels)):
    plt.subplot(len(plot_channels), 1, plot_index+1)
    plt.plot(time_downsample, signal_downsample[:, plot_channels[plot_index]])
    plt.ylabel(f"Ch {plot_channels[plot_index]+1}")
    plt.ylim(-0.1*y_max_downsample, y_max_downsample)
plt.tight_layout()
plt.show() """

""" Plot pairwise difference matrix (very large plot!)
plt.figure()
for plot_index in range(len(plot_channels)):
    for plot_diff_index in range(len(plot_channels)):
        plt.subplot(
            len(plot_channels),
            len(plot_channels),
            plot_index*len(plot_channels) + plot_diff_index + 1
        
        plt.plot(time_downsample, signal_diff[:, plot_index, plot_diff_index])
        plt.ylabel(f"Diff Ch {plot_index+1}")
        plt.ylim(-0.1*y_max_downsample, y_max_downsample)
plt.tight_layout()
plt.show() """

signal_sorted = np.sort(signal_downsample, axis=0)[::-1]
n = round(signal_sorted.shape[0] / 100)
signal_peaks = signal_sorted[:n, :]
signal_peaks_mean = np.mean(signal_peaks, axis=0)
signal_peaks_mean_sorted_index = np.argsort(signal_peaks_mean)[::-1]
signal_peaks_mean_sorted = signal_peaks_mean[signal_peaks_mean_sorted_index]

signal_correlation_matrix = np.corrcoef(signal_downsample, rowvar=False)
print("Correlation matrix done.")
high_correlation = signal_correlation_matrix > 0.9
isZero = (high_correlation == 0)
[row, col] = np.nonzero(isZero)

rectified_signal_correlation_matrix = np.corrcoef(rectified_downsampled, rowvar = False)

rectified_denoised_signal_correlation_matrix = np.corrcoef(rectified_downsampled_denoised, rowvar = False)

non_group_0 = list(signal_peaks_mean_sorted_index)
group_1 = [non_group_0[0]]
non_group_1 = []

for ch in non_group_0[1:]:
    if high_correlation[non_group_0[0], ch]:
        group_1.append(ch)
    else:
        non_group_1.append(ch)

print("Group 1:", np.array(group_1) + 1)

non_group_2 = []
group_2 = []

if len(non_group_1) > 1:
    group_2 = [non_group_1[0]]
    for ch in non_group_1[1:]:
        if high_correlation[non_group_1[0], ch]:
            group_2.append(ch)
        else:
            non_group_2.append(ch)
else:
    group_2 = non_group_1

print("Group 2:", np.array(group_2) + 1)

non_group_3 = []
group_3 = []

if len(non_group_2) > 1:
    group_3 = [non_group_2[0]]
    for ch in non_group_2[1:]:
        if high_correlation[non_group_2[0], ch]:
            group_3.append(ch)
        else:
            non_group_3.append(ch)
else:
    group_3 = non_group_2

print("Group 3:", np.array(group_3) + 1)

print("\nFinal channel groupings")
print("Group 1:", np.array(group_1) + 1)
print("Group 2:", np.array(group_2) + 1)
print("Group 3:", np.array(group_3) + 1)