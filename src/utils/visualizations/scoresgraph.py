import matplotlib.pyplot as plt
import pandas as pd
import io
import numpy as np
import re # For parsing configuration names
import argparse # For command-line arguments

def parse_config_name(config_short):
    """
    Parses the short config name into the new format:
    STLC, {Range Abbrev}, {Sim Abbrev}
    Example: knn1_range4_simc+f -> STLC, Grad, C+F
    """
    knn_part_str = "STLC" # Always STLC as per request

    range_match = re.search(r'range(\d)', config_short)
    range_abbrev = ""
    if range_match:
        range_num = range_match.group(1)
        if range_num == '1': range_abbrev = "WTA"
        elif range_num == '2': range_abbrev = "SO"
        elif range_num == '3': range_abbrev = "SA"
        elif range_num == '4': range_abbrev = "Grad"
    
    sim_part_match = re.search(r'sim([a-zA-Z+]+)$', config_short)
    sim_abbrev = ""
    if sim_part_match:
        sim_code = sim_part_match.group(1)
        if sim_code == 'a': sim_abbrev = "A"
        elif sim_code == 'c': sim_abbrev = "C"
        elif sim_code == 'c+f': sim_abbrev = "C+F"
        elif sim_code == 'm': sim_abbrev = "R"
    
    if not range_abbrev or not sim_abbrev:
        if "knn1" in config_short and not range_match and not sim_part_match: # Only knn
             return "STLC only"
        
        parts = [knn_part_str]
        if range_abbrev: parts.append(range_abbrev)
        elif range_match: # If range was found but not mapped, use raw number
            parts.append(f"R{range_match.group(1)}")

        if sim_abbrev: parts.append(sim_abbrev)
        elif sim_part_match: # If sim was found but not mapped
             parts.append(f"S_{sim_part_match.group(1)}")
        
        if len(parts) > 1:
            return ", ".join(parts)
        else: 
            return config_short.replace('knn1', 'STLC')


    # return f"{knn_part_str}, {range_abbrev}, {sim_abbrev}"
    #return f"{knn_part_str}"
    #return f"{range_abbrev}"
    return f"{sim_abbrev}"
    #return f"{range_abbrev}, {sim_abbrev}"


def plot_experimental_results(csv_file_path, output_filename="experimental_results_plot.pdf", core_f1_only=False):
    """
    Generates a multi-panel line plot from the experimental results CSV data.
    If core_f1_only is True, only plots Range F1, Similarity F1, and kNN F1.
    """
    df = pd.read_csv(csv_file_path)
    # Drop the f1_5 column as it's not needed
    if 'f1_5' in df.columns:
        df = df.drop(columns=['f1_5'])
        
    df['config_short_original'] = df['file'].apply(lambda x: x.replace('scores_', '').replace('.pkl', ''))
    df['config_display_name'] = df['config_short_original'].apply(parse_config_name)
    
    configurations = sorted(df['config_display_name'].unique())
    
    # Get unique CR values for x-axis ticks
    cr_values = sorted(df['cr'].unique())

    if core_f1_only:
        metrics_to_plot = {
            'f1_2': 'Range F1',
            'f1_3': 'Similarity F1',
            'f1_4': 'kNN F1',
        }
        nrows, ncols = 3, 1 
        figsize = (12, 21) 
        legend_bbox_anchor = (0.5, 0.04) # Adjusted for potentially larger legend
        tight_layout_rect = [0.1, 0.2, 0.9, 0.92] # Adjusted for suptitle and legend
        suptitle = 'F1 Scores vs. Compression Rate'

    else:
        metrics_to_plot = {
            'f1_1': 'Avg. F1 (Range, Sim., kNN)',
            'f1_2': 'Range Query F1',
            'f1_3': 'Similarity F1',
            'f1_4': 'kNN F1',
            'simpl_err_1': 'Error$_1$',
            'simpl_err_2': 'Error$_2$'
        }
        nrows, ncols = 3, 2 
        figsize = (18, 21) 
        legend_bbox_anchor = (0.5, 0.03) # Adjusted for potentially larger legend
        tight_layout_rect = [0, 0.25, 1, 0.92] # Adjusted for suptitle and legend
        suptitle = 'Performance Metrics vs. Compression Rate for Different Configurations'

    metric_keys = list(metrics_to_plot.keys())
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    # Ensure axes_flat is always a flat list of Axes objects
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else: # Single Axes object
        axes_flat = [axes]


    if len(configurations) <= 20:
        try:
            colors_cmap = plt.cm.get_cmap('tab20', len(configurations))
            colors = [colors_cmap(i) for i in range(len(configurations))]
        except AttributeError: 
            colors_cmap = plt.cm.viridis
            colors = [colors_cmap(i/len(configurations)) for i in range(len(configurations))]
    else: 
        colors_cmap = plt.cm.viridis
        colors = [colors_cmap(i/len(configurations)) for i in range(len(configurations))]
        
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'H', 'X', 'P'] * ( (len(configurations) // 12) + 1)

    for i, metric_key in enumerate(metric_keys):
        ax = axes_flat[i]
        plot_title = metrics_to_plot[metric_key]
        y_label = 'F1 Score' if 'f1' in metric_key else 'Simplification Error'

        for j, config_name_display in enumerate(configurations):
            subset = df[df['config_display_name'] == config_name_display].sort_values(by='cr')
            if not subset.empty:
                 ax.plot(subset['cr'], subset[metric_key], 
                        marker=markers[j % len(markers)], 
                        color=colors[j % len(colors)],
                        label=config_name_display, 
                        linewidth=2.2, markersize=8)
        
        ax.set_title(f'{plot_title} vs. CR', fontsize=30) # Further Increased
        ax.set_ylabel(y_label, fontsize=30) # Further Increased
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=26) # Further Increased
        
        # Set explicit x-ticks
        ax.set_xticks(cr_values)
        ax.set_xticklabels(cr_values, rotation=45, ha='right') # Rotate labels to prevent overlap

        if 'f1' in metric_key:
            ax.set_ylim(0, 1.05)
        if 'err' in metric_key:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

    # Set common X-axis label for the bottom-most plots
    if nrows > 0 and ncols > 0: 
        if core_f1_only: # 3x1 layout
            for k in range(ncols): 
                 axes_flat[nrows-1+k].set_xlabel('Compression Rate (CR)', fontsize=30) # Further Increased
        else: # 3x2 layout
            for k in range(ncols):
                axes_flat[(nrows-1)*ncols + k].set_xlabel('Compression Rate (CR)', fontsize=30) # Further Increased


    # Deduplicate handles and labels for the legend
    unique_labels_dict = {} 
    for ax_plot in axes_flat:
        h, l = ax_plot.get_legend_handles_labels()
        for handle, label in zip(h,l):
            if label not in unique_labels_dict:
                unique_labels_dict[label] = handle
    
    final_handles = list(unique_labels_dict.values())
    final_labels = list(unique_labels_dict.keys())
    
    num_legend_cols = 3
    if len(final_labels) > 12: num_legend_cols = 3
    elif len(final_labels) > 9: num_legend_cols = 3 
    elif len(final_labels) > 4: num_legend_cols = 2
    else: num_legend_cols = 1 if len(final_labels) > 0 else 0


    if final_handles: 
        fig.legend(final_handles, final_labels, loc='lower center', bbox_to_anchor=legend_bbox_anchor, 
                   #ncol=num_legend_cols, fontsize=28, title="Configurations (kNN)", title_fontsize=30, # Further Increased
                   #ncol=num_legend_cols, fontsize=28, title="Configurations (Range)", title_fontsize=30, # Further Increased
                   ncol=num_legend_cols, fontsize=28, title="Configurations (Similarity)", title_fontsize=30, # Further Increased
                   #ncol=num_legend_cols, fontsize=28, title="Configurations (Range, Similarity)", title_fontsize=30, # Further Increased
                   frameon=True, facecolor='white', framealpha=0.9, shadow=True)

    if not core_f1_only: # Only add suptitle if not core_f1_only
        fig.suptitle(suptitle, fontsize=40, fontweight='bold') # Further Increased
    
    current_tight_layout_rect = list(tight_layout_rect) 
    if not final_handles: 
        current_tight_layout_rect[1] = 0.02 
    
    plt.subplots_adjust(hspace=0.4 if not core_f1_only else 0.25, wspace=0.25) # Adjust spacing
    
    plt.tight_layout(rect=current_tight_layout_rect)

    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Plot saved as {output_filename}")
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plots from experimental results CSV.")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument("--output_filename", default="experimental_results_plot.pdf", help="Name of the output PDF file.")
    parser.add_argument("--core_f1_only", action="store_true", help="Only plot core F1 scores (Range, Similarity, kNN).")
    
    args = parser.parse_args()

    plot_experimental_results(args.csv_file, output_filename=args.output_filename, core_f1_only=args.core_f1_only)