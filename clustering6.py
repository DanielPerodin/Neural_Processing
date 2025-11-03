import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neo.io import NeuralynxIO
import tridesclous as tdc
import json
from datetime import datetime
import tempfile
import shutil
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

print(f"Using tridesclous version: {tdc.__version__}")

def calculate_cluster_quality_metrics(waveforms, spike_times, sampling_rate, cluster_label):
    """Calculate quality metrics for a specific cluster"""
    if len(waveforms) == 0:
        return {
            "cluster_label": int(cluster_label),
            "n_spikes": 0, 
            "firing_rate": 0, 
            "spikes_per_minute": 0,
            "snr": 0, 
            "isolation_quality": 0,
            "median_peak_to_trough_amplitude": 0
        }
    
    waveforms_array = np.array(waveforms)
    
    # Signal-to-noise ratio
    template = np.median(waveforms_array, axis=0)
    signal_amplitude = np.abs(np.min(template))
    
    residuals = []
    for wf in waveforms_array:
        residual = wf - template
        residuals.extend(residual)
    
    noise_std = np.std(residuals) if residuals else 1e-10
    snr = signal_amplitude / noise_std
    
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
    
    return {
        "cluster_label": int(cluster_label),
        "n_spikes": len(spike_times),
        "firing_rate": float(firing_rate),
        "spikes_per_minute": float(spikes_per_minute),
        "snr": float(snr),
        "isolation_quality": float(isolation_quality),
        "median_peak_to_trough_amplitude": float(median_peak_to_trough_amplitude)
    }

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
    
    # Plot 3: SNR
    axes[0, 2].bar(cluster_labels, [m['snr'] for m in cluster_metrics_list], color=colors)
    axes[0, 2].set_xlabel("Cluster Label")
    axes[0, 2].set_ylabel("SNR")
    axes[0, 2].set_title("Signal-to-Noise Ratio")
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
    summary_text = f"Channel {channel_id} Summary\n"
    summary_text += "="*30 + "\n\n"
    for m in cluster_metrics_list:
        summary_text += f"Cluster {m['cluster_label']}:\n"
        summary_text += f"  Spikes: {m['n_spikes']}\n"
        summary_text += f"  Rate: {m['firing_rate']:.2f} Hz\n"
        summary_text += f"  SNR: {m['snr']:.2f}\n\n"
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
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
    Returns cluster templates, spike times, and quality metrics.
    """
    print(f"\n{'='*60}")
    print(f"Processing Channel {channel_id}")
    print(f"{'='*60}")
    
    temp_folder = tempfile.mkdtemp(prefix=f"tdc_channel_{channel_id}_")

    try:
        # Extract single channel data
        channel_data = raw_data[:, channel_id:channel_id+1].astype('float32')
        duration_sec = len(channel_data) / sampling_rate
        
        # Save raw data
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
        
        if n_clusters == 0:
            print(f"⚠️  No clusters found for channel {channel_id}, but checking for detected spikes...")
            
            # Check if there are any spikes (even if unclustered/noise)
            all_spike_indices = spikes['index']
            
            if len(all_spike_indices) == 0:
                print(f"No spikes detected at all for channel {channel_id}")
                return None
            
            print(f"Found {len(all_spike_indices)} detected spikes (unclustered)")
            
            # Extract all detected waveforms
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
                print(f"No valid waveforms extracted for channel {channel_id}")
                return None
            
            all_waveforms = np.array(all_waveforms)
            
            # Calculate mean template of all detected spikes
            mean_template = np.mean(all_waveforms, axis=0)
            
            # Save the "no cluster" template with clear naming
            templates_file = os.path.join(output_dir, f'channel_{channel_id}_NO_CLUSTERS_detected_spikes_template.npy')
            np.save(templates_file, {'no_cluster_template': mean_template})
            print(f"No-cluster template saved: {templates_file}")
            
            # Calculate basic metrics for all detected spikes
            metrics = calculate_cluster_quality_metrics(all_waveforms, all_spike_times, sampling_rate, -1)
            metrics['cluster_label'] = 'NO_CLUSTERS'
            
            metrics_file = os.path.join(output_dir, f'channel_{channel_id}_NO_CLUSTERS_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Create visualization
            t_axis = np.linspace(-pre_samples / sampling_rate * 1000, 
                                post_samples / sampling_rate * 1000, 
                                n_template_samples)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"Channel {channel_id} — NO CLUSTERS FOUND ({len(all_waveforms)} spikes detected)", 
                        fontsize=16, fontweight='bold', color='orange')
            
            # Plot 1: All detected waveforms + mean
            ax = axes[0, 0]
            n_to_plot = min(200, len(all_waveforms))
            indices_to_plot = np.random.choice(len(all_waveforms), n_to_plot, replace=False)
            
            for i in indices_to_plot:
                ax.plot(t_axis, all_waveforms[i], color='lightgray', linewidth=0.5, alpha=0.3)
            
            ax.plot(t_axis, mean_template, color='red', linewidth=3, 
                   label=f'Mean Template ({len(all_waveforms)} spikes)')
            ax.set_xlabel("Time (ms)", fontsize=12)
            ax.set_ylabel("Amplitude (µV)", fontsize=12)
            ax.set_title("Detected Spikes (Unclustered)", fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Spike time histogram
            ax = axes[0, 1]
            ax.hist(all_spike_times, bins=min(50, len(all_spike_times)//5 + 1),
                   alpha=0.7, color='gray', edgecolor='black')
            ax.set_xlabel("Time (seconds)", fontsize=12)
            ax.set_ylabel("Spike count", fontsize=12)
            ax.set_title("Spike Times Distribution", fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: PCA 2D (if we have enough spikes)
            ax = axes[1, 0]
            if len(all_waveforms) >= 3:
                try:
                    pca = PCA(n_components=min(2, len(all_waveforms)))
                    pca_2d = pca.fit_transform(all_waveforms)
                    ax.scatter(pca_2d[:, 0], pca_2d[:, 1], color='gray', s=10, alpha=0.5)
                    ax.set_xlabel("Principal Component 1", fontsize=12)
                    ax.set_ylabel("Principal Component 2", fontsize=12)
                    ax.set_title("2D PCA (No Clear Clusters)", fontweight='bold')
                    ax.grid(True, alpha=0.3)
                except:
                    ax.text(0.5, 0.5, "PCA failed\n(insufficient variance)", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
            else:
                ax.text(0.5, 0.5, f"Not enough spikes for PCA\n({len(all_waveforms)} spikes)", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
            
            # Plot 4: Metrics summary
            ax = axes[1, 1]
            ax.axis('off')
            metrics_text = f"Channel {channel_id} — NO CLUSTERS\n"
            metrics_text += "="*40 + "\n\n"
            metrics_text += f"Total Spikes Detected: {len(all_waveforms)}\n"
            metrics_text += f"Firing Rate: {metrics['firing_rate']:.2f} Hz\n"
            metrics_text += f"SNR: {metrics['snr']:.2f}\n"
            metrics_text += f"Isolation Quality: {metrics['isolation_quality']:.3f}\n"
            metrics_text += f"Peak-to-Trough: {metrics['median_peak_to_trough_amplitude']:.1f} µV\n\n"
            metrics_text += "⚠️  Tridesclous could not identify\n"
            metrics_text += "distinct clusters in this data.\n"
            metrics_text += "This may indicate:\n"
            metrics_text += "  • High noise levels\n"
            metrics_text += "  • Poor signal quality\n"
            metrics_text += "  • No clear units present\n"
            metrics_text += "  • Insufficient spike count"
            
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=11, fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.9))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"channel_{channel_id}_NO_CLUSTERS.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ No-cluster visualization saved for channel {channel_id}")
            
            # Return a result indicating no clusters but spikes detected
            return {
                'channel_id': channel_id,
                'n_clusters': 0,
                'n_spikes_detected': len(all_waveforms),
                'cluster_templates': {'no_cluster': mean_template},
                'cluster_metrics': [metrics],
                'status': 'NO_CLUSTERS_DETECTED'
            }
        
        print(f"Found {n_clusters} clusters on channel {channel_id}")
        
        # Extract waveforms and organize by cluster
        n_template_samples = catalogue['centers0'].shape[1]
        pre_samples = n_template_samples // 2
        post_samples = n_template_samples - pre_samples
        
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
        
        # Calculate mean templates for each cluster
        cluster_templates = {}
        cluster_metrics = []
        
        for label in unique_labels:
            waveforms = cluster_data[label]['waveforms']
            times = cluster_data[label]['times']
            
            # Calculate mean template
            mean_template = np.mean(waveforms, axis=0)
            cluster_templates[label] = mean_template
            
            # Calculate quality metrics
            metrics = calculate_cluster_quality_metrics(waveforms, times, sampling_rate, label)
            cluster_metrics.append(metrics)
            
            print(f"  Cluster {label}: {len(waveforms)} spikes, {metrics['firing_rate']:.2f} Hz")
        
        # Save cluster templates
        templates_file = os.path.join(output_dir, f'channel_{channel_id}_cluster_templates.npy')
        np.save(templates_file, cluster_templates)
        print(f"Cluster templates saved: {templates_file}")
        
        # Save quality metrics
        metrics_file = os.path.join(output_dir, f'channel_{channel_id}_cluster_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(cluster_metrics, f, indent=2)
        print(f"Cluster metrics saved: {metrics_file}")
        
        # Create quality metrics plot
        save_cluster_metrics_plot(channel_id, cluster_metrics, output_dir)
        
        # Perform PCA
        all_waveforms = np.array(all_waveforms)
        all_labels = np.array(all_labels)
        
        print("Performing PCA on waveforms...")
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(all_waveforms)
        
        # Generate visualizations
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        # PLOT 1: Mean Cluster Waveforms
        plt.figure(figsize=(16, 9))
        t_axis = np.linspace(-pre_samples / sampling_rate * 1000, 
                            post_samples / sampling_rate * 1000, 
                            n_template_samples)
        
        # Plot individual waveforms (sample)
        for label in unique_labels:
            mask = (all_labels == label)
            wfs_to_plot = all_waveforms[mask]
            
            n_to_plot = min(200, len(wfs_to_plot))
            indices_to_plot = np.random.choice(len(wfs_to_plot), n_to_plot, replace=False)
            
            for i in indices_to_plot:
                plt.plot(t_axis, wfs_to_plot[i], color=color_map[label], 
                        linewidth=0.5, alpha=0.15)
        
        # Plot mean templates
        for label in unique_labels:
            mask = (all_labels == label)
            mean_template = cluster_templates[label]
            n_spikes = np.sum(mask)
            fr = cluster_metrics[label]['firing_rate']
            plt.plot(t_axis, mean_template, color=color_map[label], linewidth=3, zorder=10,
                    label=f"Cluster {label} ({n_spikes} spikes, {fr:.1f} Hz)")
        
        plt.xlabel("Time (ms)", fontsize=14)
        plt.ylabel("Amplitude (µV)", fontsize=14)
        plt.title(f"Channel {channel_id} — Mean Cluster Waveforms", fontsize=16, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"channel_{channel_id}_clusters_mean.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # PLOT 2: 2D PCA
        fig, ax = plt.subplots(figsize=(10, 8))
        for label in unique_labels:
            mask = (all_labels == label)
            ax.scatter(principal_components[mask, 0], principal_components[mask, 1],
                      color=color_map[label], s=15, alpha=0.7, label=f"Cluster {label}")
        
        ax.set_xlabel("Principal Component 1", fontsize=12)
        ax.set_ylabel("Principal Component 2", fontsize=12)
        ax.set_title(f"Channel {channel_id} — 2D PCA of Clusters", fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"channel_{channel_id}_pca_2d.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # PLOT 3: 3D PCA
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for label in unique_labels:
            mask = (all_labels == label)
            ax.scatter(principal_components[mask, 0], principal_components[mask, 1], 
                      principal_components[mask, 2],
                      color=color_map[label], s=15, alpha=0.6, label=f"Cluster {label}")
        
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        ax.set_title(f"Channel {channel_id} — 3D PCA of Clusters", fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        plt.savefig(os.path.join(output_dir, f"channel_{channel_id}_pca_3d.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save spike times per cluster
        for label in unique_labels:
            spike_times = cluster_data[label]['times']
            df = pd.DataFrame({
                'channel_id': channel_id,
                'cluster_label': label,
                'spike_time': spike_times
            })
            csv_file = os.path.join(output_dir, f'channel_{channel_id}_cluster_{label}_spike_times.csv')
            df.to_csv(csv_file, index=False)
            print(f"Spike times saved: {csv_file}")
        
        return {
            'channel_id': channel_id,
            'n_clusters': n_clusters,
            'cluster_templates': cluster_templates,
            'cluster_metrics': cluster_metrics,
            'cluster_data': cluster_data
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
            # Check if all channels are covered
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
    
    print(f"\n{'='*60}")
    print(f"APPLYING CAR")
    print(f"{'='*60}")
    
    for i, group in enumerate(car_groups):
        if len(group) == 1:
            # Individual channel - subtract its own mean
            ch = group[0]
            car_reference = raw_data[:, ch:ch+1].mean(axis=1, keepdims=True)
            car_data[:, ch:ch+1] = raw_data[:, ch:ch+1] - car_reference
            print(f"Group {i+1}: Channel {ch} (individual CAR)")
        else:
            # Multiple channels - subtract group mean
            car_reference = raw_data[:, group].mean(axis=1, keepdims=True)
            for ch in group:
                car_data[:, ch:ch+1] = raw_data[:, ch:ch+1] - car_reference
            print(f"Group {i+1}: Channels {group} (shared CAR, {len(group)} channels)")
    
    print("CAR applied successfully.")
    return car_data

def process_all_channels(raw_data, sampling_rate, excluded_channels, output_dir='clustering_results'):
    """Process all channels with Tridesclous clustering"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    n_channels = raw_data.shape[1]
    included_channels = [ch for ch in range(n_channels) if ch not in excluded_channels]
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {len(included_channels)} CHANNELS")
    print(f"{'='*60}")
    print(f"Excluded channels: {sorted(excluded_channels)}")
    print(f"Processing channels: {included_channels}")
    
    all_results = {}
    
    for channel_id in included_channels:
        result = cluster_channel_with_tridesclous(channel_id, raw_data, sampling_rate, output_dir)
        if result is not None:
            all_results[channel_id] = result
    
    # Save summary
    summary = {
        'n_channels_processed': len(all_results),
        'total_channels': n_channels,
        'excluded_channels': excluded_channels,
        'included_channels': included_channels,
        'sampling_rate': float(sampling_rate),
        'duration_sec': float(raw_data.shape[0] / sampling_rate),
        'timestamp': datetime.now().isoformat(),
        'channels': {}
    }
    
    for channel_id, result in all_results.items():
        summary['channels'][str(channel_id)] = {
            'n_clusters': result['n_clusters'],
            'cluster_metrics': result['cluster_metrics']
        }
    
    summary_file = os.path.join(output_dir, 'processing_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nProcessing summary saved: {summary_file}")
    
    return all_results

# -------------------------
# MAIN PIPELINE
# -------------------------

# Load Neuralynx Data
data_folder = r"C:\path\to\the\data"

reader = NeuralynxIO(dirname=data_folder)
blk = reader.read_block(lazy=False)
signal = blk.segments[0].analogsignals[0]
sampling_rate = signal.sampling_rate.rescale('Hz').item()
raw_data = np.array(signal)

print(f"Data shape: {raw_data.shape}, Sampling rate: {sampling_rate} Hz")

# Step 1: Get channels to exclude
excluded_channels = get_excluded_channels(raw_data.shape[1])

# Step 2: Get CAR groupings
car_groups = get_car_groups(raw_data.shape[1], excluded_channels)

# Step 3: Apply CAR
car_data = apply_car_to_data(raw_data, car_groups)

# Step 4: Process all channels
all_results = process_all_channels(car_data, sampling_rate, excluded_channels, 
                                   output_dir='clustering_results')

print("\n" + "="*60)
print("PROCESSING COMPLETE")
print("="*60)
print(f"Successfully processed {len(all_results)} channels")
print(f"Results saved in: clustering_results/")