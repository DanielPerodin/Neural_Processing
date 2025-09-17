import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compare_templates_across_sessions(template_files, output_dir='template_comparison'): 
    """
    Compare neuron templates across multiple recording sessions to track channels over time.

    Parameters:
    - template_files: list of paths to .npy template files from different sessions
    - output_dir: directory to save comparison plots and correlation matrix

    Returns:
    - correlation_results: cross-session template correlations
    - stability_scores: detailed analysis of template stability
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n{'='*60}")
    print("TEMPLATE COMPARISON ACROSS SESSIONS")
    print(f"{'='*60}")

    # Load all template files
    all_templates = {}
    session_names = []

    for i, template_file in enumerate(template_files, start=1):
        if os.path.exists(template_file):
            templates = np.load(template_file, allow_pickle=True).item()
            session_name = f"Session_{i}"
            all_templates[session_name] = templates
            session_names.append(session_name)
            print(f"Loaded {session_name} from file: {os.path.basename(template_file)} "
                  f"({len(templates)} channels)")
        else:
            print(f"Warning: Template file not found: {template_file}")

    if len(all_templates) < 2:
        print("Error: Need at least 2 template files for comparison")
        return None, None

    # Collect all unique channel IDs across sessions
    all_channels = set()
    for templates in all_templates.values():
        all_channels.update(templates.keys())
    all_channels = sorted(list(all_channels))

    print(f"Channels found across sessions: {all_channels}")

    correlation_results = {}
    stability_scores = {}

    # Compare each channel across sessions
    for channel in all_channels:
        print(f"\nAnalyzing channel {channel}:")
        channel_templates = {}

        # Gather this channel's template from each session
        for session_name in session_names:
            if channel in all_templates[session_name]:
                template = all_templates[session_name][channel]

                # If template is 2D (samples x channels), pick this channel's column
                if template.ndim == 2:
                    if channel < template.shape[1]:
                        primary_waveform = template[:, channel]
                    else:
                        # fallback to mean across columns
                        primary_waveform = template.mean(axis=1)
                else:
                    primary_waveform = template  # already 1-D

                channel_templates[session_name] = primary_waveform

        # Only compare if channel appears in at least two sessions
        if len(channel_templates) >= 2:
            correlations = {}

            for i, session1 in enumerate(channel_templates.keys()):
                for session2 in list(channel_templates.keys())[i+1:]:
                    corr = np.corrcoef(channel_templates[session1],
                                       channel_templates[session2])[0, 1]
                    correlations[f"{session1}_vs_{session2}"] = corr
                    print(f"  {session1} vs {session2}: r = {corr:.3f}")

            mean_correlation = np.mean(list(correlations.values()))
            stability_scores[channel] = {
                'mean_correlation': mean_correlation,
                'correlations': correlations,
                'n_sessions': len(channel_templates)
            }

            # Plot overlaid templates and correlation matrix
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            colors = plt.cm.Set1(np.linspace(0, 1, len(channel_templates)))
            time_axis = np.arange(len(next(iter(channel_templates.values()))))

            for (session_name, waveform), color in zip(channel_templates.items(), colors):
                ax1.plot(time_axis, waveform, label=session_name,
                         color=color, linewidth=2)

            ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Sample index')
            ax1.set_ylabel('Amplitude (μV)')
            ax1.set_title(f'Channel {channel} Templates Across Sessions')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Correlation matrix
            n_sessions = len(channel_templates)
            corr_matrix = np.ones((n_sessions, n_sessions))
            session_list = list(channel_templates.keys())

            for i, session1 in enumerate(session_list):
                for j, session2 in enumerate(session_list):
                    if i != j:
                        corr_matrix[i, j] = np.corrcoef(channel_templates[session1],
                                                       channel_templates[session2])[0, 1]

            im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax2.set_xticks(range(n_sessions))
            ax2.set_yticks(range(n_sessions))
            ax2.set_xticklabels(session_list, rotation=45)
            ax2.set_yticklabels(session_list)
            ax2.set_title(f'Channel {channel} Correlation Matrix\n(Mean r = {mean_correlation:.3f})')

            # Annotate values
            for i in range(n_sessions):
                for j in range(n_sessions):
                    ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black")

            plt.colorbar(im, ax=ax2)
            plt.tight_layout()

            plot_file = os.path.join(output_dir, f'channel_{channel}_comparison.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            correlation_results[channel] = correlations
        else:
            print(f"  Not enough sessions for comparison ({len(channel_templates)} sessions)")

    # Summary JSON
    summary_file = os.path.join(output_dir, 'template_stability_report.json')
    summary_data = {
        'session_files': template_files,
        'session_names': session_names,
        'channels_analyzed': list(stability_scores.keys()),
        'stability_scores': stability_scores,
        'correlation_results': correlation_results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)

    print(f"\nTemplate comparison complete! Results saved to: {output_dir}/")
    print(f"Summary report: template_stability_report.json")

    # Print stability summary
    print(f"\n{'='*40}")
    print("TEMPLATE STABILITY SUMMARY")
    print(f"{'='*40}")
    for channel in sorted(stability_scores.keys()):
        score = stability_scores[channel]
        level = "HIGH" if score['mean_correlation'] > 0.8 else \
                "MEDIUM" if score['mean_correlation'] > 0.5 else "LOW"
        print(f"Channel {channel:2d}: r = {score['mean_correlation']:.3f} ({level} stability)")

    return correlation_results, stability_scores


compare_templates_across_sessions([
   # Place template files as shown below:
   #  r'C:\path\to\session_x.npy',
   #  r'C:\path\to\session_y.npy',
   #  r'C:\path\to\session_z.npy',
], output_dir='template_comparison')