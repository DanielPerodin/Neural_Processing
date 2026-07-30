import os
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# this is based on all of the sampling rates from Mouse 36 and 46, 
# will change to be based on reading the sampling rate from data later
SAMPLING_RATE = 32000 

def plot_overlay(entries, channel_name, session_color_map, output_path, suffix=""):
    """Helper to generate the overlay plot with ms on x-axis."""
    if not entries:
        print(f"No templates to plot for {channel_name} {suffix}")
        return

    fig_wave, ax_wave = plt.subplots(figsize=(12, 5))
    
    # Calculate time axis in ms
    sample_count = len(entries[0][1])
    time_axis_ms = (np.arange(sample_count) / SAMPLING_RATE) * 1000

    for label, wave in entries:
        session_part = label.split(" / ")[0]
        cluster_part = label.split(" / ")[1]
        # Robust parsing of cluster index for styling
        try:
            cluster_idx = int(cluster_part.split("_")[1])
        except:
            cluster_idx = 0
            
        base_color = session_color_map.get(session_part, 'black')
        alpha = 1.0 - 0.25 * (cluster_idx % 4)
        lw = 2.5 - 0.4 * (cluster_idx % 4)
        ls = ['-', '--', ':', '-.'][cluster_idx % 4]

        ax_wave.plot(time_axis_ms, wave, label=label, color=base_color, 
                     alpha=max(alpha, 0.4), linewidth=lw, linestyle=ls)

    ax_wave.set_title(f'Templates: {channel_name} {suffix}')
    ax_wave.set_xlabel('Time (ms)')
    ax_wave.set_ylabel('Amplitude ($\mu$V)')
    ax_wave.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.15, 1))
    ax_wave.grid(True, alpha=0.3)
    fig_wave.tight_layout()
    
    fig_wave.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig_wave)

def compare_templates_across_sessions(folder_paths, output_dir='template_comparison'):
    os.makedirs(output_dir, exist_ok=True)
    rej_dir = os.path.join(output_dir, 'rejected_templates')

    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found – {folder_path}")
            continue

        channel_name = os.path.basename(os.path.normpath(folder_path))
        template_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
        
        if len(template_files) < 2:
            print(f"Skipping {channel_name}: Need at least 2 .npy files for comparison.")
            continue

        print(f"\n{'='*60}\nPROCESSING CHANNEL: {channel_name}\n{'='*60}")

        # 1. Load Data
        all_sessions = {}
        session_names = []
        for i, path in enumerate(template_files, start=1):
            raw = np.load(path, allow_pickle=True)
            templates = raw.item() if raw.dtype == object else {0: raw}
            
            base_filename = os.path.splitext(os.path.basename(path))[0]
            marker = "templates"
            idx = base_filename.lower().find(marker)
            name = f"Session_{base_filename[idx+len(marker):]}" if idx != -1 else f"Session_{i}"
            
            all_sessions[name] = templates
            session_names.append(name)

        # 2. Build entries list
        entries = []
        for session_name in session_names:
            templates = all_sessions[session_name]
            for cluster_id in sorted(templates.keys()):
                if cluster_id == 'metadata': continue
                waveform = templates[cluster_id]
                if waveform.ndim == 2:
                    col = cluster_id if cluster_id < waveform.shape[1] else 0
                    waveform = waveform[:, col]
                label = f"{session_name} / Cluster_{cluster_id}"
                entries.append((label, waveform.astype(float)))

        # 3. Setup styling
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(session_names)))
        session_color_map = {name: colors[i] for i, name in enumerate(session_names)}

        # 4. Initial Plot
        wave_path = os.path.join(output_dir, f'{channel_name}_overlay.png')
        plot_overlay(entries, channel_name, session_color_map, wave_path)
        print(f"Initial overlay saved to: {wave_path}")

        # 5. Handle Template Rejection
        print("\n--- Template Rejection ---")
        print("Look at the generated plot. Type labels to reject (comma-separated).")
        print("Example: Session_29 / Cluster_0, Session_30 / Cluster_1")
        reject_input = input("Reject labels (or press ENTER to keep all): ")
        
        if reject_input.strip():
            rejected_list = [r.strip() for r in reject_input.split(',')]
            os.makedirs(rej_dir, exist_ok=True)
            
            final_entries = []
            for label, wave in entries:
                if label in rejected_list:
                    # Save to rejected folder
                    rej_path = os.path.join(rej_dir, f"{channel_name}_{label.replace(' / ', '_')}.npy")
                    np.save(rej_path, wave)
                    print(f"Rejected and stored: {label}")
                else:
                    final_entries.append((label, wave))
            
            # Generate Filtered Plot
            filtered_path = os.path.join(output_dir, f'{channel_name}_overlay_filtered.png')
            plot_overlay(final_entries, channel_name, session_color_map, filtered_path, suffix="(Filtered)")
            print(f"Filtered overlay saved to: {filtered_path}")
            
            # Update entries for matrices/JSON to the filtered set
            entries = final_entries
        
        # 6. Correlation Matrices & JSON (using the final/filtered set)
        n = len(entries)
        if n < 2:
            print(f"Not enough templates left in {channel_name} to generate matrices.")
            continue

        labels = [e[0] for e in entries]
        waves = [e[1] for e in entries]
        corr_matrix = np.ones((n, n))
        for i, j in itertools.product(range(n), repeat=2):
            if i != j:
                corr_matrix[i, j] = np.corrcoef(waves[i], waves[j])[0, 1]

        # Save Matrix Plot
        fig_m, ax_m = plt.subplots(figsize=(8, 6))
        im = ax_m.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax_m.set_title(f'Correlation: {channel_name}')
        plt.colorbar(im)
        fig_m.savefig(os.path.join(output_dir, f'{channel_name}_matrices.png'))
        plt.close(fig_m)

        # Save JSON
        with open(os.path.join(output_dir, f'{channel_name}_report.json'), 'w') as f:
            json.dump({'channel': channel_name, 'labels': labels, 'corr': corr_matrix.tolist()}, f, indent=2)

if __name__ == '__main__':
    # Configuration
    folders = [
        # r'C:\path\to\Channel_14',
        # r'C:\path\to\Channel_15',
        
        
       
       

        
    ]
    compare_templates_across_sessions(folders)