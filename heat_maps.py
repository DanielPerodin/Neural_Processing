import re
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load json files - sorted by name

files = glob.glob("json_files/*.json")

def extract_day_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

files = sorted(files, key=extract_day_number)

if len(files) == 0:
    raise ValueError("No JSON files found in directory.")

print(f"Found {len(files)} files.")
print(files)

# collect data

data_amp = {}        
data_clusters = {}  

all_channels = set()
num_days = len(files)

for day_index, file in enumerate(files):

    day = day_index + 1  

    with open(file, "r") as f:
        data = json.load(f)

    for ch_id, ch_data in data["channels"].items():
        ch_id = int(ch_id)
        all_channels.add(ch_id)

        n_clusters = ch_data["n_clusters"]

        if ch_id not in data_clusters:
            data_clusters[ch_id] = {}

        data_clusters[ch_id][day] = n_clusters

        for cluster in ch_data["cluster_metrics"]:
            amp = cluster["median_peak_to_trough_amplitude"]

            if ch_id not in data_amp:
                data_amp[ch_id] = {}

            data_amp[ch_id][day] = amp

# bulid matrices

all_channels = sorted(all_channels)
all_days = list(range(1, num_days + 1))

amp_matrix = np.full((len(all_channels), num_days), np.nan)
cluster_matrix = np.full((len(all_channels), num_days), np.nan)

for i, ch in enumerate(all_channels):
    for j, day in enumerate(all_days):

        if ch in data_amp and day in data_amp[ch]:
            amp_matrix[i, j] = data_amp[ch][day]

        if ch in data_clusters and day in data_clusters[ch]:
            cluster_matrix[i, j] = data_clusters[ch][day]

amplitude_df = pd.DataFrame(
    amp_matrix,
    index=[f"Channel {ch}" for ch in all_channels],
    columns=all_days
)

clusters_df = pd.DataFrame(
    cluster_matrix,
    index=[f"Channel {ch}" for ch in all_channels],
    columns=all_days
)

print("Amplitude Data:")
print(amplitude_df)

print("\nCluster Count Data:")
print(clusters_df)

# plot amplitude heatmap

amplitude_df = amplitude_df.fillna(0)

plt.figure(figsize=(12, 6))
plt.imshow(amplitude_df, aspect="auto", cmap="autumn_r", vmin = 0)
plt.colorbar(label="Median Peak-to-Trough Amplitude")

plt.xticks(range(num_days), [i * 7 for i in range(num_days)])
plt.yticks(range(len(all_channels)), amplitude_df.index)

plt.xlabel("Days Elapsed")
plt.ylabel("Channel")
plt.title("Peak-to-Trough Amplitude Across Days")

plt.tight_layout()
plt.show()

# plot cluster heat map

clusters_df = clusters_df.fillna(0)

plt.figure(figsize=(12, 6))
plt.imshow(clusters_df, aspect="auto", cmap="autumn_r", vmin = 0)
plt.colorbar(label="Number of Clusters")

plt.xticks(range(num_days), [i * 7 for i in range(num_days)])
plt.yticks(range(len(all_channels)), clusters_df.index)

plt.xlabel("Days Elapsed")
plt.ylabel("Channel")
plt.title("Cluster Count Across Days")

# add numbers inside cells
for i in range(len(all_channels)):
    for j in range(num_days):
        value = clusters_df.iloc[i, j]
        if not np.isnan(value):
            plt.text(j, i, int(value), ha='center', va='center', color='black')

plt.tight_layout()
plt.show()