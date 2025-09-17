import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from neo.io import NeuralynxIO
import tridesclous as tdc
import json
from datetime import datetime

print(f"Using tridesclous version: {tdc.__version__}")

def save_plot_as_png(fig, filename, rejected=False):
    """Save matplotlib figure as PNG with rejection indicator"""
    if rejected:
        filename = filename.replace('.png', '_REJECTED.png')
    
    # Add timestamp to title
    if fig.axes:
        current_title = fig.axes[0].get_title()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if rejected:
            new_title = f"{current_title} [REJECTED - {timestamp}]"
            fig.suptitle(new_title, color='red', fontweight='bold')
        else:
            new_title = f"{current_title} [ACCEPTED - {timestamp}]"
            fig.suptitle(new_title, color='green', fontweight='bold')
    
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {filename}")

def calculate_quality_metrics(waveforms, spike_times, sampling_rate):
    """Calculate quality metrics for a channel"""
    if len(waveforms) == 0:
        return {
            "n_spikes": 0, 
            "firing_rate": 0, 
            "spikes_per_minute": 0,
            "snr": 0, 
            "isolation_quality": 0,
            "median_peak_to_trough_amplitude": 0
        }
    
    waveforms_array = np.array(waveforms)
    
    # Signal-to-noise ratio (proper electrophysiology calculation)
    # SNR = peak amplitude / noise standard deviation
    template = np.median(waveforms_array, axis=0)
    primary_ch = np.argmax(np.abs(template).max(axis=0))  # Channel with largest signal
    
    # Signal: peak amplitude of the template on primary channel
    signal_amplitude = np.abs(np.min(template[:, primary_ch]))  # Peak amplitude (absolute value)
    
    # Noise: standard deviation of residuals from template
    residuals = []
    for wf in waveforms_array:
        residual = wf[:, primary_ch] - template[:, primary_ch]
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
    
    # Simple isolation quality (coefficient of variation of waveform amplitudes)
    amplitudes = [np.min(wf) for wf in waveforms_array]
    if len(amplitudes) > 1:
        isolation_quality = np.std(amplitudes) / (np.abs(np.mean(amplitudes)) + 1e-10)
    else:
        isolation_quality = 0
    
    # Median peak-to-trough amplitude
    peak_to_trough_amplitudes = []
    for wf in waveforms_array:
        # Use the primary channel (assume it's the one with largest amplitude)
        primary_ch = np.argmax(np.abs(wf).max(axis=0))
        wf_primary = wf[:, primary_ch]
        
        # Find peak (most negative) and trough (most positive after peak)
        peak_idx = np.argmin(wf_primary)
        peak_amplitude = wf_primary[peak_idx]  # Most negative value
        
        if peak_idx < len(wf_primary) - 1:
            # Find the trough (most positive value after the peak)
            trough_amplitude = np.max(wf_primary[peak_idx:])
            # Peak-to-trough amplitude is the difference (always positive)
            peak_to_trough_amp = trough_amplitude - peak_amplitude
            peak_to_trough_amplitudes.append(peak_to_trough_amp)
    
    median_peak_to_trough_amplitude = np.median(peak_to_trough_amplitudes) if peak_to_trough_amplitudes else 0
    
    return {
        "n_spikes": len(spike_times),
        "firing_rate": firing_rate,
        "spikes_per_minute": spikes_per_minute,
        "snr": snr,
        "isolation_quality": isolation_quality,
        "median_peak_to_trough_amplitude": median_peak_to_trough_amplitude
    }

def detect_spikes_and_extract_features(data, sampling_rate, excluded_channels=None):
    """Perform spike detection and feature extraction"""
    print("Performing spike detection...")
    
    if excluded_channels is None:
        excluded_channels = []
    
    # Only process channels that are NOT excluded
    included_channels = [ch for ch in range(data.shape[1]) if ch not in excluded_channels]
    print(f"Processing channels: {included_channels} (excluding {excluded_channels})")
    
    # Simple threshold-based spike detection
    threshold = -5 * np.median(np.abs(data), axis=0) / 0.6745  # MAD-based threshold
    
    all_spikes = []
    spike_times = []
    spike_waveforms = []
    channel_spike_data = {}  # Store per-channel data
    
    # Detect spikes only on included channels
    for ch in included_channels:
        signal_ch = data[:, ch]
        
        # Find peaks below threshold
        spike_indices = []
        for i in range(25, len(signal_ch) - 40):  # avoid edges
            if (signal_ch[i] < threshold[ch] and 
                signal_ch[i] < signal_ch[i-1] and 
                signal_ch[i] < signal_ch[i+1]):
                # Check if this is a local minimum in a larger window
                window_start = max(0, i-10)
                window_end = min(len(signal_ch), i+11)
                if signal_ch[i] == np.min(signal_ch[window_start:window_end]):
                    spike_indices.append(i)
        
        # Remove spikes too close together (refractory period)
        refractory_samples = int(0.001 * sampling_rate)  # 1ms refractory period
        cleaned_spikes = []
        last_spike = -refractory_samples
        
        for spike_idx in spike_indices:
            if spike_idx - last_spike > refractory_samples:
                cleaned_spikes.append(spike_idx)
                last_spike = spike_idx
        
        print(f"Channel {ch}: {len(cleaned_spikes)} spikes detected")
        
        # Extract waveforms and times for this channel
        channel_waveforms = []
        channel_times = []
        
        for spike_idx in cleaned_spikes:
            if spike_idx >= 25 and spike_idx < len(signal_ch) - 40:
                waveform = data[spike_idx-25:spike_idx+40, :]  # all channels
                spike_waveforms.append(waveform)
                channel_waveforms.append(waveform)
                spike_time = spike_idx / sampling_rate
                spike_times.append(spike_time)
                channel_times.append(spike_time)
                all_spikes.append([ch, spike_time])
        
        # Store channel-specific data
        channel_spike_data[ch] = {
            'waveforms': channel_waveforms,
            'times': channel_times,
            'indices': cleaned_spikes
        }
    
    print(f"Total spikes detected: {len(spike_times)}")
    
    # Create templates only for included channels
    templates = {}
    quality_metrics = {}
    
    for ch in included_channels:
        ch_data = channel_spike_data[ch]
        if ch_data['waveforms']:
            templates[ch] = np.median(ch_data['waveforms'], axis=0)
            quality_metrics[ch] = calculate_quality_metrics(
                ch_data['waveforms'], ch_data['times'], sampling_rate)
        else:
            quality_metrics[ch] = calculate_quality_metrics([], [], sampling_rate)
    
    return all_spikes, templates, quality_metrics, channel_spike_data

def run_spike_sorting_pipeline(raw_data, sampling_rate, excluded_channels=None, run_name="initial"):
    """Run the complete spike sorting pipeline"""
    print(f"\n{'='*60}")
    print(f"RUNNING SPIKE SORTING PIPELINE - {run_name.upper()}")
    print(f"{'='*60}")
    
    if excluded_channels is None:
        excluded_channels = []
    
    # Create results directory
    results_dir = f'results_{run_name}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    print(f"Results will be saved to: {results_dir}/")
    
    # Apply CAR excluding rejected channels
    if excluded_channels:
        print(f"Excluding channels {excluded_channels} from CAR calculation...")
        included_channels = [ch for ch in range(raw_data.shape[1]) if ch not in excluded_channels]
        car_reference = raw_data[:, included_channels].mean(axis=1, keepdims=True)
        car_data = raw_data - car_reference
        print(f"CAR applied using {len(included_channels)}/{raw_data.shape[1]} channels")
    else:
        print("Applying CAR using all channels...")
        car_data = raw_data - raw_data.mean(axis=1, keepdims=True)
        print(f"CAR applied using all {raw_data.shape[1]} channels")
    
    # Detect spikes and extract features
    all_spikes, templates, quality_metrics, channel_spike_data = detect_spikes_and_extract_features(
        car_data, sampling_rate, excluded_channels)
    
    # Save results
    results = {
        'run_name': run_name,
        'timestamp': datetime.now().isoformat(),
        'excluded_channels': excluded_channels,
        'total_channels': raw_data.shape[1],
        'recording_duration_sec': raw_data.shape[0] / sampling_rate,
        'sampling_rate': sampling_rate,
        'quality_metrics': quality_metrics
    }
    
    # Export CSV (only include non-excluded channels)
    if excluded_channels:
        filtered_spikes = [spike for spike in all_spikes if spike[0] not in excluded_channels]
    else:
        filtered_spikes = all_spikes
    
    df = pd.DataFrame(filtered_spikes, columns=['neuron_id', 'spike_time'])
    csv_file = os.path.join(results_dir, f'spike_times_{run_name}.csv')
    df.to_csv(csv_file, index=False)
    print(f"CSV saved: {csv_file} with {len(df)} spikes")
    results['exported_spikes'] = len(df)
    
    # Save templates
    templates_file = os.path.join(results_dir, f'neuron_templates_{run_name}.npy')
    np.save(templates_file, templates)
    print(f"Templates saved: {templates_file} ({len(templates)} templates)")
    
    # Save quality metrics
    metrics_file = os.path.join(results_dir, f'quality_metrics_{run_name}.json')
    with open(metrics_file, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json_metrics = {}
        for ch, metrics in quality_metrics.items():
            json_metrics[str(ch)] = convert_numpy_types(metrics)
        
        json_data = {
            'results': convert_numpy_types(results), 
            'channel_metrics': json_metrics
        }
        json.dump(json_data, f, indent=2)
    print(f"Quality metrics saved: {metrics_file}")
    
    return all_spikes, templates, quality_metrics, channel_spike_data, results_dir

def create_enhanced_qc_interface(templates, quality_metrics, channel_spike_data, 
                                all_spikes, results_dir, excluded_channels_input=None):
    """Enhanced quality control interface with accept/reject functionality"""
    
    if len(templates) == 0:
        print("No templates found for visualization")
        return []
    
    # Track decisions
    accepted_channels = set()
    rejected_channels = set(excluded_channels_input) if excluded_channels_input else set()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    plt.subplots_adjust(bottom=0.15, top=0.9)
    
    channel_ids = list(templates.keys())
    current_idx = 0
    
    def plot_channel(idx):
        if idx >= len(channel_ids):
            return
            
        channel_id = channel_ids[idx]
        
        # Clear all axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        # Get channel data
        channel_data = channel_spike_data.get(channel_id, {'waveforms': [], 'times': []})
        channel_spikes = channel_data['waveforms']
        channel_times = channel_data['times']
        metrics = quality_metrics.get(channel_id, {})
        
        # Get template for this channel
        template = templates.get(channel_id, None)
        
        # Determine channel status
        status = "REJECTED" if channel_id in rejected_channels else "ACCEPTED" if channel_id in accepted_channels else "PENDING"
        status_color = {'REJECTED': 'red', 'ACCEPTED': 'green', 'PENDING': 'orange'}[status]
        
        if len(channel_spikes) > 0 and template is not None:
            # Create time axis in milliseconds
            # Waveforms are extracted from -25 to +40 samples around spike peak
            n_samples = template.shape[0]  # Use template shape instead of indexing
            time_before_peak_ms = 25 / sampling_rate * 1000  # convert to ms
            time_after_peak_ms = 40 / sampling_rate * 1000   # convert to ms
            time_axis_ms = np.linspace(-time_before_peak_ms, time_after_peak_ms, n_samples)
            
            # Plot 1: Individual waveforms on this channel only
            sample_spikes = channel_spikes[:min(100, len(channel_spikes))]
            for wf in sample_spikes:
                ax1.plot(time_axis_ms, wf[:, channel_id], color='lightblue', alpha=0.3, linewidth=0.8)
            
            # Plot template for this channel
            ax1.plot(time_axis_ms, template[:, channel_id], color='red', linewidth=3, label='Template')
            ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Spike Peak')
            ax1.set_title(f"Channel {channel_id} Waveforms [{status}]", color=status_color, fontweight='bold')
            ax1.set_xlabel("Time (ms)")
            ax1.set_ylabel("Amplitude (μV)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: All channels for this template (to see cross-channel activity)
            for ch in range(template.shape[1]):
                offset = ch * 100
                ax2.plot(time_axis_ms, template[:, ch] + offset, 
                        color='red' if ch == channel_id else 'gray', 
                        linewidth=3 if ch == channel_id else 1,
                        label=f'Ch {ch}' if ch == channel_id else '')
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_title(f"Template Across All Channels (Channel {channel_id} highlighted)")
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("Channel + offset (μV)")
            if channel_id == channel_id:  # Always true, but keeps structure
                ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Spike times histogram
            if channel_times:
                ax3.hist(channel_times, bins=min(50, len(channel_times)//5 + 1), 
                        alpha=0.7, color=status_color, edgecolor='black')
                ax3.set_title(f"Spike Times Distribution")
                ax3.set_xlabel("Time (seconds)")
                ax3.set_ylabel("Spike count")
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Quality metrics display (simplified, no recommendations)
            metrics_text = f"QUALITY METRICS - Channel {channel_id}\n"
            metrics_text += f"Status: {status}\n"
            metrics_text += "="*30 + "\n"
            metrics_text += f"Spike Count: {metrics.get('n_spikes', 0)}\n"
            metrics_text += f"Firing Rate: {metrics.get('firing_rate', 0):.2f} Hz\n"
            metrics_text += f"Spikes/Minute: {metrics.get('spikes_per_minute', 0):.1f}\n"
            metrics_text += f"Signal-to-Noise: {metrics.get('snr', 0):.2f}\n"
            metrics_text += f"Isolation Quality: {metrics.get('isolation_quality', 0):.3f}\n"
            metrics_text += f"Peak-to-Trough: {metrics.get('median_peak_to_trough_amplitude', 0):.1f} μV"
            
            ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontsize=11, fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.9))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.set_xticks([])
            ax4.set_yticks([])
            ax4.set_title("Quality Metrics", fontweight='bold')
        else:
            # No spikes found
            for ax, title in zip([ax1, ax2, ax3, ax4], 
                               ["No Waveforms", "No Template", "No Spike Times", "No Data"]):
                ax.text(0.5, 0.5, f"Channel {channel_id}: No spikes detected", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(title)
        
        plt.tight_layout()
        fig.canvas.draw_idle()
        
        # Save plot
        plot_filename = os.path.join(results_dir, f'channel_{channel_id:02d}_qc.png')
        save_plot_as_png(fig, plot_filename, rejected=(channel_id in rejected_channels))
    
    # Initial plot
    plot_channel(current_idx)
    
    # Create controls
    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.2, 0.02, 0.4, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, 'Channel', 0, len(channel_ids)-1, 
                   valinit=0, valstep=1, valfmt='%d')
    
    def update(val):
        plot_channel(int(slider.val))
    
    slider.on_changed(update)
    
    # Buttons
    ax_accept = plt.axes([0.65, 0.02, 0.08, 0.04])
    ax_reject = plt.axes([0.75, 0.02, 0.08, 0.04])
    ax_save_all = plt.axes([0.85, 0.02, 0.08, 0.04])
    
    button_accept = Button(ax_accept, 'Accept', color='lightgreen')
    button_reject = Button(ax_reject, 'Reject', color='lightcoral')
    button_save_all = Button(ax_save_all, 'Save All', color='lightblue')
    
    def accept_channel(event):
        current_channel = channel_ids[int(slider.val)]
        accepted_channels.add(current_channel)
        if current_channel in rejected_channels:
            rejected_channels.remove(current_channel)
        print(f"Channel {current_channel} ACCEPTED")
        plot_channel(int(slider.val))  # Refresh plot
    
    def reject_channel(event):
        current_channel = channel_ids[int(slider.val)]
        rejected_channels.add(current_channel)
        if current_channel in accepted_channels:
            accepted_channels.remove(current_channel)
        print(f"Channel {current_channel} REJECTED")
        plot_channel(int(slider.val))  # Refresh plot
    
    def save_all_plots(event):
        print("Saving all channel plots...")
        current_idx_backup = int(slider.val)
        for i in range(len(channel_ids)):
            plot_channel(i)
            plt.pause(0.1)  # Brief pause to ensure plot updates
        plot_channel(current_idx_backup)  # Return to current
        print("All plots saved!")
    
    button_accept.on_clicked(accept_channel)
    button_reject.on_clicked(reject_channel)
    button_save_all.on_clicked(save_all_plots)
    
    plt.show()
    
    return list(rejected_channels)

# -------------------------
# MAIN PIPELINE
# -------------------------

# 1. Load Neuralynx Data
data_folder = r"C:\path\to\folder"

reader = NeuralynxIO(dirname=data_folder)
blk = reader.read_block(lazy=False)
signal = blk.segments[0].analogsignals[0]
sampling_rate = signal.sampling_rate.rescale('Hz').item()
raw_data = np.array(signal)

print(f"Data shape: {raw_data.shape}, Sampling rate: {sampling_rate} Hz")

# 2. Initial run with all channels
print("\nINITIAL ANALYSIS - Using all channels for CAR")
all_spikes_initial, templates_initial, quality_metrics_initial, channel_spike_data_initial, results_dir_initial = run_spike_sorting_pipeline(
    raw_data, sampling_rate, excluded_channels=None, run_name="initial"
)

# 3. Quality Control Interface
print("\nSTARTING QUALITY CONTROL")
print("Instructions:")
print("- Use the slider to browse through channels")
print("- Click 'Accept' for good channels, 'Reject' for bad ones") 
print("- Click 'Save All' to save all channel plots as PNG files")
print("- Quality metrics and recommendations are shown in the bottom-right panel")
print("- All plots are automatically saved when you browse or make decisions")
print("- Close the window when you're done with QC")

rejected_channels = create_enhanced_qc_interface(
    templates_initial, quality_metrics_initial, channel_spike_data_initial, 
    all_spikes_initial, results_dir_initial
)

# 4. Offer to re-run without rejected channels
if rejected_channels:
    print(f"\nCHANNELS REJECTED: {sorted(rejected_channels)}")
    print("\nSince CAR was computed using ALL channels (including the rejected ones),")
    print("the rejected channels may have contaminated the common average reference.")
    print("\nWould you like to re-run the analysis excluding the rejected channels?")
    print("This will recompute CAR using only the accepted channels.")
    
    response = input("\nRe-run analysis without rejected channels? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print(f"\nRE-RUNNING ANALYSIS - Excluding channels {sorted(rejected_channels)}")
        all_spikes_clean, templates_clean, quality_metrics_clean, channel_spike_data_clean, results_dir_clean = run_spike_sorting_pipeline(
            raw_data, sampling_rate, excluded_channels=rejected_channels, run_name="clean"
        )
        
        print(f"\nCLEAN ANALYSIS COMPLETE")
        print(f"Results saved to: {results_dir_clean}/")
        print("\nWould you like to run QC on the clean data as well?")
        
        qc_response = input("Run QC on clean data? (y/n): ").lower().strip()
        if qc_response in ['y', 'yes']:
            print("\nQUALITY CONTROL - CLEAN DATA")
            rejected_channels_clean = create_enhanced_qc_interface(
                templates_clean, quality_metrics_clean, channel_spike_data_clean, 
                all_spikes_clean, results_dir_clean, excluded_channels_input=rejected_channels
            )
    else:
        print("\nUsing initial results with all channels.")
else:

    print("\nNO CHANNELS REJECTED - Initial analysis is complete!")
