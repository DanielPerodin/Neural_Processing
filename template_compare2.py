import os
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def compare_templates_across_sessions(folder_paths, output_dir='template_comparison'):
    """
    Compare neuron templates across multiple recording sessions, organized by folders (channels).

    Parameters
    ----------
    folder_paths : list[str]
        Paths to directories, each containing .npy template files for a specific channel.
    output_dir : str
        Directory where plots and the JSON report are saved.
    """

    os.makedirs(output_dir, exist_ok=True)

    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found – {folder_path}")
            continue

        # Use the folder name as the Channel Name
        channel_name = os.path.basename(os.path.normpath(folder_path))
        
        # Get all .npy files in this folder
        template_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
        
        if len(template_files) < 2:
            print(f"Skipping {channel_name}: Need at least 2 .npy files for comparison.")
            continue

        print(f"\n{'='*60}")
        print(f"PROCESSING CHANNEL: {channel_name}")
        print(f"{'='*60}")

        # ------------------------------------------------------------------ #
        # 1.  Load all sessions within this folder                           #
        # ------------------------------------------------------------------ #
        all_sessions = {}
        session_names = []

        for i, path in enumerate(template_files, start=1):
            raw = np.load(path, allow_pickle=True)
            templates = raw.item() if raw.dtype == object else {0: raw}

            # Extract filename and apply the "templates" suffix logic
            base_filename = os.path.splitext(os.path.basename(path))[0]
            marker = "templates"
            idx = base_filename.lower().find(marker)
            
            if idx != -1:
                suffix = base_filename[idx + len(marker):]
                name = f"Session_{suffix}"
            else:
                name = f"Session_{i}"

            all_sessions[name] = templates
            session_names.append(name)

        # ------------------------------------------------------------------ #
        # 2.  Build flat list of (label, waveform)                           #
        # ------------------------------------------------------------------ #
        entries = []
        for session_name in session_names:
            templates = all_sessions[session_name]
            for cluster_id in sorted(templates.keys()):
                waveform = templates[cluster_id]
                if waveform.ndim == 2:
                    col = cluster_id if cluster_id < waveform.shape[1] else 0
                    waveform = waveform[:, col]
                label = f"{session_name} / Cluster_{cluster_id}"
                entries.append((label, waveform.astype(float)))

        n = len(entries)
        labels = [e[0] for e in entries]
        waves = [e[1] for e in entries]

        # ------------------------------------------------------------------ #
        # 3.  Correlation & Covariance matrices                              #
        # ------------------------------------------------------------------ #
        corr_matrix = np.ones((n, n))
        cov_matrix = np.zeros((n, n))

        for i, j in itertools.product(range(n), repeat=2):
            if i == j:
                corr_matrix[i, j] = 1.0
                cov_matrix[i, j] = np.cov(waves[i], waves[i])[0, 1]
            else:
                c = np.corrcoef(waves[i], waves[j])
                corr_matrix[i, j] = c[0, 1]
                cov_matrix[i, j] = np.cov(waves[i], waves[j])[0, 1]

        # ------------------------------------------------------------------ #
        # 4.  Waveform overlay plot                                          #
        # ------------------------------------------------------------------ #
        session_base_colors = plt.cm.tab10(np.linspace(0, 0.9, len(session_names)))
        session_color_map = {name: session_base_colors[i] for i, name in enumerate(session_names)}

        fig_wave, ax_wave = plt.subplots(figsize=(12, 5))
        time_axis = np.arange(len(waves[0]))

        for label, wave in entries:
            session_part = label.split(" / ")[0]
            cluster_part = label.split(" / ")[1]
            cluster_idx = int(cluster_part.split("_")[1])
            base_color = session_color_map[session_part]
            alpha = 1.0 - 0.25 * cluster_idx
            lw = 2.5 - 0.4 * cluster_idx
            ls = ['-', '--', ':', '-.'][cluster_idx % 4]

            ax_wave.plot(time_axis, wave, label=label, color=base_color, 
                         alpha=max(alpha, 0.5), linewidth=lw, linestyle=ls)

        ax_wave.set_title(f'Templates: {channel_name}')
        ax_wave.legend(fontsize=8, loc='upper right')
        ax_wave.grid(True, alpha=0.3)
        fig_wave.tight_layout()
        
        wave_path = os.path.join(output_dir, f'{channel_name}_overlay.png')
        fig_wave.savefig(wave_path, dpi=300, bbox_inches='tight')
        plt.close(fig_wave)

        # ------------------------------------------------------------------ #
        # 5.  Matrices Plot                                                  #
        # ------------------------------------------------------------------ #
        fig, axes = plt.subplots(1, 2, figsize=(7 * n * 0.55 + 4, n * 0.55 + 4))
        fig.suptitle(f'Matrices: {channel_name}', fontsize=14, fontweight='bold', y=1.01)

        short_labels = [f"{lbl.split(' / ')[0]}\n{lbl.split(' / ')[1]}" for lbl in labels]
        cell_font = max(5, 10 - n)

        # Correlation
        ax_corr = axes[0]
        im_corr = ax_corr.imshow(corr_matrix, cmap='coolwarm', norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
        ax_corr.set_xticks(range(n)); ax_corr.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        ax_corr.set_yticks(range(n)); ax_corr.set_yticklabels(short_labels, fontsize=8)
        ax_corr.set_title('Correlation')
        plt.colorbar(im_corr, ax=ax_corr, shrink=0.8)

        # Covariance
        ax_cov = axes[1]
        max_abs_cov = np.max(np.abs(cov_matrix))
        im_cov = ax_cov.imshow(cov_matrix, cmap='PuOr', norm=TwoSlopeNorm(vmin=-max_abs_cov, vcenter=0, vmax=max_abs_cov))
        ax_cov.set_xticks(range(n)); ax_cov.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        ax_cov.set_yticks(range(n)); ax_cov.set_yticklabels(short_labels, fontsize=8)
        ax_cov.set_title('Covariance')
        plt.colorbar(im_cov, ax=ax_cov, shrink=0.8)

        fig.tight_layout()
        matrix_path = os.path.join(output_dir, f'{channel_name}_matrices.png')
        fig.savefig(matrix_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # ------------------------------------------------------------------ #
        # 6.  JSON Report                                                    #
        # ------------------------------------------------------------------ #
        report = {
            'channel': channel_name,
            'entry_labels': labels,
            'correlation_matrix': corr_matrix.tolist(),
        }
        report_path = os.path.join(output_dir, f'{channel_name}_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Finished {channel_name}. Saved plots and report.")


if __name__ == '__main__':
    # List your folder paths here
    folders_to_process = [
        # r'C:\path\to\Channel_14',
        # r'C:\path\to\Channel_15',
        r'C:\Users\perod\WaveletMatching\mouse46\Ch2',
        r'C:\Users\perod\WaveletMatching\mouse46\Ch3',
        r'C:\Users\perod\WaveletMatching\mouse46\Ch4',
        r'C:\Users\perod\WaveletMatching\mouse46\Ch5',
        r'C:\Users\perod\WaveletMatching\mouse46\Ch6',
        r'C:\Users\perod\WaveletMatching\mouse46\Ch7',
        r'C:\Users\perod\WaveletMatching\mouse46\Ch9',
        r'C:\Users\perod\WaveletMatching\mouse46\Ch10',
        r'C:\Users\perod\WaveletMatching\mouse46\Ch11',
        r'C:\Users\perod\WaveletMatching\mouse46\Ch12',
        r'C:\Users\perod\WaveletMatching\mouse46\Ch13',
        r'C:\Users\perod\WaveletMatching\mouse46\Ch14',
        r'C:\Users\perod\WaveletMatching\mouse46\Ch15',
        
    ]
    compare_templates_across_sessions(folders_to_process, output_dir='template_comparison')