import matplotlib.pyplot as plt
import pandas as pd
import io
import numpy as np
import re # For parsing configuration names

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
        elif sim_code == 'm': sim_abbrev = "M"
    
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
    return f"{range_abbrev}, {sim_abbrev}"


def plot_experimental_results(data_csv, output_filename="experimental_results_plot.pdf", core_f1_only=False):
    """
    Generates a multi-panel line plot from the experimental results CSV data.
    If core_f1_only is True, only plots Range F1, Similarity F1, and kNN F1.
    """
    df = pd.read_csv(io.StringIO(data_csv))
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
        legend_bbox_anchor = (0.5, 0.02)
        tight_layout_rect = [0.1, 0.08, 0.9, 0.95] 
        suptitle = 'F1 Scores vs. Compression Rate' 

    else:
        metrics_to_plot = {
            'f1_1': 'Avg. F1 (Range, Sim., kNN)',
            'f1_2': 'Range F1',
            'f1_3': 'Similarity F1',
            'f1_4': 'kNN F1',
            'simpl_err_1': 'Error$_1$',
            'simpl_err_2': 'Error$_2$'
        }
        nrows, ncols = 3, 2 
        figsize = (18, 21) 
        legend_bbox_anchor = (0.5, 0.01)
        tight_layout_rect = [0, 0.06, 1, 0.95] 
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
        
        ax.set_title(f'{plot_title} vs. CR', fontsize=16)
        ax.set_ylabel(y_label, fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Set explicit x-ticks
        ax.set_xticks(cr_values)
        ax.set_xticklabels(cr_values) # Ensure labels match the ticks

        if 'f1' in metric_key:
            ax.set_ylim(0, 1.05)
        if 'err' in metric_key:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

    # Set common X-axis label for the bottom-most plots
    if nrows > 0 and ncols > 0: 
        if core_f1_only: # 3x1 layout
            for k in range(ncols): 
                 axes_flat[nrows-1+k].set_xlabel('Compression Rate (CR)', fontsize=14)
        else: # 3x2 layout
            for k in range(ncols):
                axes_flat[(nrows-1)*ncols + k].set_xlabel('Compression Rate (CR)', fontsize=14)


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
    if len(final_labels) > 12: num_legend_cols = 4
    elif len(final_labels) > 9: num_legend_cols = 3 
    elif len(final_labels) > 4: num_legend_cols = 2
    else: num_legend_cols = 1 if len(final_labels) > 0 else 0


    if final_handles: 
        fig.legend(final_handles, final_labels, loc='lower center', bbox_to_anchor=legend_bbox_anchor, 
                   ncol=num_legend_cols, fontsize=11, title="Configurations (Range, Similarity)", title_fontsize=13,
                   frameon=True, facecolor='white', framealpha=0.9, shadow=True)

    if not core_f1_only: # Only add suptitle if not core_f1_only
        fig.suptitle(suptitle, fontsize=20, fontweight='bold')
    
    current_tight_layout_rect = list(tight_layout_rect) 
    if not final_handles: 
        current_tight_layout_rect[1] = 0.02 
    
    plt.subplots_adjust(hspace=0.4 if not core_f1_only else 0.25, wspace=0.25) # Adjust spacing
    
    plt.tight_layout(rect=current_tight_layout_rect)

    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Plot saved as {output_filename}")
    # plt.show()

# --- Your CSV Data as a String ---
data = """
folder,file,cr,f1_1,f1_2,f1_3,f1_4,f1_5,simpl_err_1,simpl_err_2
1_1_1_a,scores_knn1_range1_sima.pkl,0.8,0.6927380952380954,0.96,0.32154761904761914,0.7966666666666667,0.0,11372.385255846984,4074.967241458396
1_1_1_a,scores_knn1_range1_sima.pkl,0.9,0.6652380952380953,0.97,0.2457142857142857,0.7800000000000001,0.0,12135.388537344665,4236.361913247862
1_1_1_a,scores_knn1_range1_sima.pkl,0.95,0.6807222222222221,0.98,0.3121666666666667,0.7499999999999997,0.0,12833.807673227746,4424.900115832937
1_1_1_a,scores_knn1_range1_sima.pkl,0.975,0.5303977873977874,0.94,0.3278600288600289,0.3233333333333334,0.0,13037.204339742646,4495.13594961778
1_1_1_a,scores_knn1_range1_sima.pkl,0.99,0.46928401169298867,0.7053368835638146,0.1791818181818182,0.5233333333333331,0.0,16076.176041485392,5302.099826970657
8_1_3_c+f,scores_knn1_range3_simc+f.pkl,0.8,0.47023373730899964,0.5444631166889038,0.2862380952380953,0.5799999999999998,0.0,17932.48372557378,5939.960572397217
8_1_3_c+f,scores_knn1_range3_simc+f.pkl,0.9,0.4203453050994102,0.5463312369356575,0.15137134502923977,0.5633333333333332,0.0,18405.458178317967,6094.4196526486385
8_1_3_c+f,scores_knn1_range3_simc+f.pkl,0.95,0.4462487302311225,0.5249611029740691,0.26045175438596496,0.5533333333333332,0.0,18599.68026790794,6144.82579441258
8_1_3_c+f,scores_knn1_range3_simc+f.pkl,0.975,0.3718236741741456,0.4584710225224365,0.18033333333333335,0.47666666666666685,0.0,18844.828045818813,6159.320370233338
8_1_3_c+f,scores_knn1_range3_simc+f.pkl,0.99,0.420107639419876,0.4183562515929616,0.22863333333333333,0.6133333333333332,0.0,18925.490161388236,6161.287251354047
4_1_2_a,scores_knn1_range2_sima.pkl,0.8,0.642748332172819,0.8910069012803621,0.34390476190476194,0.6933333333333332,0.0,14840.284047212084,5000.779386243107
4_1_2_a,scores_knn1_range2_sima.pkl,0.9,0.5625177858742698,0.7834296773814822,0.2674570135746607,0.6366666666666664,0.0,16017.998067415683,5438.018093093816
4_1_2_a,scores_knn1_range2_sima.pkl,0.95,0.4882136966908587,0.6559484493799355,0.2586926406926407,0.55,0.0,16732.495215460804,5624.435126331848
4_1_2_a,scores_knn1_range2_sima.pkl,0.975,0.4646045367284272,0.6180502624219341,0.22243001443001448,0.5533333333333332,0.0,17132.139148601615,5732.873942660517
4_1_2_a,scores_knn1_range2_sima.pkl,0.99,0.28270534171637085,0.4470604695935569,0.1577222222222222,0.2433333333333334,0.0,17668.944867772872,5770.781830853965
6_1_3_c,scores_knn1_range3_simc.pkl,0.8,0.4516247915562415,0.5675569143512642,0.22398412698412704,0.5633333333333335,0.0,17932.081863025654,5939.847853906927
6_1_3_c,scores_knn1_range3_simc.pkl,0.9,0.4326166914672771,0.5019198934716502,0.23593018093018092,0.56,0.0,18405.222338096566,6094.309452809446
6_1_3_c,scores_knn1_range3_simc.pkl,0.95,0.44815160086909545,0.5236254375279215,0.17416269841269844,0.6466666666666664,0.0,18599.60142622116,6145.286222439901
6_1_3_c,scores_knn1_range3_simc.pkl,0.975,0.416936637315853,0.44433372147136857,0.2531428571428571,0.5533333333333335,0.0,18844.80763854415,6159.24354555148
6_1_3_c,scores_knn1_range3_simc.pkl,0.99,0.41964394129793087,0.49036616732813615,0.24523232323232316,0.5233333333333333,0.0,18925.495120528427,6161.393975645343
10_1_4_a,scores_knn1_range4_sima.pkl,0.8,0.47908377688310927,0.5527001040981012,0.2745512265512266,0.61,0.0,17848.649517393307,5923.309878026505
10_1_4_a,scores_knn1_range4_sima.pkl,0.9,0.4599479819649946,0.5206888787399169,0.32915506715506715,0.5299999999999998,0.0,18420.409168305894,6093.543515428773
10_1_4_a,scores_knn1_range4_sima.pkl,0.95,0.39306843386287166,0.43090859829191175,0.19496336996336996,0.5533333333333332,0.0,18622.100895414318,6163.888978482517
10_1_4_a,scores_knn1_range4_sima.pkl,0.975,0.28662287107482926,0.4223581237139982,0.1841771561771562,0.2533333333333333,0.0,18790.15134416383,6142.8650349183
10_1_4_a,scores_knn1_range4_sima.pkl,0.99,0.29601588364541914,0.4173809842695909,0.13733333333333334,0.33333333333333326,0.0,18946.779225045088,6161.8365444718265
11_1_4_c+f,scores_knn1_range4_simc+f.pkl,0.8,0.47938226425888275,0.5571561722860278,0.26765728715728715,0.6133333333333332,0.0,17847.91987295856,5923.250756826651
11_1_4_c+f,scores_knn1_range4_simc+f.pkl,0.9,0.43002074972388515,0.48556247701398875,0.21116643882433356,0.5933333333333333,0.0,18420.62781706166,6093.587506475041
11_1_4_c+f,scores_knn1_range4_simc+f.pkl,0.95,0.41520116199775065,0.4618250977148638,0.23711172161172162,0.5466666666666665,0.0,18622.394778739344,6163.926020494406
11_1_4_c+f,scores_knn1_range4_simc+f.pkl,0.975,0.44463252932191505,0.5049019169700745,0.20899567099567096,0.6199999999999997,0.0,18790.385860558115,6142.979616133589
11_1_4_c+f,scores_knn1_range4_simc+f.pkl,0.99,0.3794738684771341,0.391057639687002,0.17736396574440053,0.57,0.0,18947.444011956402,6161.886013556567
9_1_4_c,scores_knn1_range4_simc.pkl,0.8,0.4579894374135251,0.5218630188411642,0.2821052933994111,0.5700000000000002,0.0,17848.55817543484,5922.747485625669
9_1_4_c,scores_knn1_range4_simc.pkl,0.9,0.4061391436798532,0.5016208942430229,0.18346320346320344,0.5333333333333333,0.0,18420.564184731735,6093.736795897929
9_1_4_c,scores_knn1_range4_simc.pkl,0.95,0.40693836113701093,0.5073700284659778,0.18344505494505495,0.53,0.0,18622.230091033558,6165.476700341554
9_1_4_c,scores_knn1_range4_simc.pkl,0.975,0.2940355846590761,0.43265615555604164,0.22611726508785338,0.22333333333333333,0.0,18789.780719174618,6143.030741367176
9_1_4_c,scores_knn1_range4_simc.pkl,0.99,0.30401207231148125,0.46289985329808014,0.1458030303030303,0.3033333333333333,0.0,18947.71495099355,6161.979941994001
7_1_3_a,scores_knn1_range3_sima.pkl,0.8,0.497138675672741,0.5792804958826919,0.3021355311355312,0.61,0.0,17932.1333118203,5939.8674361684
7_1_3_a,scores_knn1_range3_sima.pkl,0.9,0.3810620844805171,0.45339188114717893,0.14979437229437229,0.54,0.0,18405.171718867663,6094.184220798052
7_1_3_a,scores_knn1_range3_sima.pkl,0.95,0.4157229444249499,0.4358643599703763,0.28797113997114,0.5233333333333333,0.0,18599.617221224515,6144.810896305777
7_1_3_a,scores_knn1_range3_sima.pkl,0.975,0.3709228429696723,0.5025430387129384,0.24689215686274515,0.36333333333333334,0.0,18844.776464700582,6159.19759838806
7_1_3_a,scores_knn1_range3_sima.pkl,0.99,0.2987201777945635,0.4871605333836906,0.17566666666666667,0.23333333333333325,0.0,18925.466816141194,6161.321557189595
5_1_2_c+f,scores_knn1_range2_simc+f.pkl,0.8,0.6192312670201656,0.8552493566160523,0.2657777777777778,0.7366666666666667,0.0,14848.265349257164,4997.922843985497
5_1_2_c+f,scores_knn1_range2_simc+f.pkl,0.9,0.5698926253157616,0.7199096010934833,0.3364349415204678,0.6533333333333333,0.0,16048.98877586811,5434.349937439754
5_1_2_c+f,scores_knn1_range2_simc+f.pkl,0.95,0.5270046258306907,0.730836388314583,0.3035108225108225,0.5466666666666665,0.0,16756.257447541,5622.250278929918
5_1_2_c+f,scores_knn1_range2_simc+f.pkl,0.975,0.48637444035481037,0.6334752879381375,0.235648033126294,0.5899999999999997,0.0,17159.936407509595,5727.550725908077
5_1_2_c+f,scores_knn1_range2_simc+f.pkl,0.99,0.5216699741775379,0.6295207450434362,0.33215584415584415,0.603333333333333,0.0,17692.21097777234,5762.738617042713
3_1_2_c,scores_knn1_range2_simc.pkl,0.8,0.6237026241813068,0.8883294842655322,0.2761117216117216,0.7066666666666666,0.0,14597.253498013673,4974.283548469269
3_1_2_c,scores_knn1_range2_simc.pkl,0.9,0.5493018824509899,0.709038980686303,0.25220000000000004,0.6866666666666665,0.0,15706.471976881176,5415.899874728988
3_1_2_c,scores_knn1_range2_simc.pkl,0.95,0.5130666486565177,0.6837127664823739,0.2554871794871795,0.6,0.0,16407.176717346276,5592.1662156330885
3_1_2_c,scores_knn1_range2_simc.pkl,0.975,0.37504035350423076,0.6780734414650732,0.21038095238095242,0.2366666666666667,0.0,16804.266079705703,5700.957451892733
3_1_2_c,scores_knn1_range2_simc.pkl,0.99,0.38106120944415833,0.6162788664277132,0.1769047619047619,0.3500000000000001,0.0,17368.719442883394,5740.616067413441
"""

if __name__ == '__main__':
    plot_experimental_results(data, output_filename="exp_results_all_nogauss.pdf", core_f1_only=True)