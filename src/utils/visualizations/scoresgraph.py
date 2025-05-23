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
    
    # Extract similarity part more robustly
    # It's usually the last part after the final underscore, if underscores exist after "rangeX_"
    sim_part_match = re.search(r'sim([a-zA-Z+]+)$', config_short)
    sim_abbrev = ""
    if sim_part_match:
        sim_code = sim_part_match.group(1)
        if sim_code == 'a': sim_abbrev = "A"
        elif sim_code == 'c': sim_abbrev = "C"
        elif sim_code == 'c+f': sim_abbrev = "C+F"
        elif sim_code == 'm': sim_abbrev = "M"
    
    # Fallback if parsing fails to extract parts, return original short name
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


    return f"{knn_part_str}, {range_abbrev}, {sim_abbrev}"


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

    if core_f1_only:
        metrics_to_plot = {
            'f1_2': 'Range F1',
            'f1_3': 'Similarity F1',
            'f1_4': 'kNN F1',
        }
        nrows, ncols = 3, 1  # Changed to stack vertically
        figsize = (12, 21)  # Adjusted for vertical layout
        legend_bbox_anchor = (0.5, -0.02)  # Adjusted for vertical layout
        tight_layout_rect = [0.1, 0.05, 0.9, 0.95]  # Added more space on sides
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
        nrows, ncols = 6, 1  # Changed to stack vertically
        figsize = (12, 42)  # Adjusted for vertical layout
        legend_bbox_anchor = (0.5, -0.02)  # Adjusted for vertical layout
        tight_layout_rect = [0.1, 0.05, 0.9, 0.95]  # Added more space on sides
        suptitle = 'Performance Metrics vs. Compression Rate for Different Configurations'

    metric_keys = list(metrics_to_plot.keys())
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    # Ensure axes is always a 2D array for consistent indexing, even if nrows=1
    if nrows == 1 and ncols > 1:
        axes_flat = axes
    elif nrows > 1 and ncols == 1:
        axes_flat = axes
    elif nrows == 1 and ncols == 1:
        axes_flat = [axes] # Make it a list containing the single Axes object
    else:
        axes_flat = axes.flatten()


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
        if 'f1' in metric_key:
            ax.set_ylim(0, 1.05)
        if 'err' in metric_key:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

    # Set common X-axis label
    if nrows > 1:  # For vertical layout
        for ax in axes_flat:
            ax.set_xlabel('Compression Rate (CR)', fontsize=14)
    else:  # Single plot
        axes_flat[0].set_xlabel('Compression Rate (CR)', fontsize=14)


    # Deduplicate handles and labels for the legend
    unique_labels_dict = {} # Use a dict to preserve order of first appearance
    for ax_plot in axes_flat:
        h, l = ax_plot.get_legend_handles_labels()
        for handle, label in zip(h,l):
            if label not in unique_labels_dict:
                unique_labels_dict[label] = handle
    
    final_handles = list(unique_labels_dict.values())
    final_labels = list(unique_labels_dict.keys())
    
    # Sort legend items by label for consistency, if needed, though order of appearance is often good.
    # sorted_legend = sorted(zip(final_labels, final_handles), key=lambda x: x[0])
    # if sorted_legend:
    #     final_labels, final_handles = zip(*sorted_legend)


    num_legend_cols = 3
    if len(final_labels) > 12: num_legend_cols = 4
    elif len(final_labels) > 9: num_legend_cols = 3 # Adjusted for better fit
    elif len(final_labels) > 4: num_legend_cols = 2
    else: num_legend_cols = 1 if len(final_labels) > 0 else 0


    if final_handles: 
        fig.legend(final_handles, final_labels, loc='lower center', bbox_to_anchor=legend_bbox_anchor, 
                   ncol=num_legend_cols, fontsize=11, title="Configurations (kNN, Range, Similarity)", title_fontsize=13,
                   frameon=True, facecolor='white', framealpha=0.9, shadow=True)

    fig.suptitle(suptitle, fontsize=20, fontweight='bold')
    
    # Adjust rect based on whether legend is present for bottom margin
    current_tight_layout_rect = list(tight_layout_rect) # Make it a list to modify
    if not final_handles: # No legend, less bottom margin needed
        current_tight_layout_rect[1] = 0.02 
    
    # Add more space between subplots
    plt.subplots_adjust(hspace=0.3)  # Increase vertical space between subplots
    
    plt.tight_layout(rect=current_tight_layout_rect)

    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Plot saved as {output_filename}")
    # plt.show()

# --- Your CSV Data as a String ---
data = """folder,file,cr,f1_1,f1_2,f1_3,f1_4,f1_5,simpl_err_1,simpl_err_2
14,scores_knn1_range4_simc+f.pkl,0.8,0.6917893868262935,0.8007736099383775,0.6679278838738365,0.6066666666666667,0.0,812188.7565414518,507218.29663247237
14,scores_knn1_range4_simc+f.pkl,0.9,0.4130589438124365,0.544633639938721,0.33454319149858863,0.3600000000000001,0.0,816677.7222328881,507813.74110176286
14,scores_knn1_range4_simc+f.pkl,0.95,0.26611328989298444,0.3517083943262494,0.22329814201937068,0.22333333333333327,0.0,817426.3098995466,503994.14859924396
14,scores_knn1_range4_simc+f.pkl,0.975,0.17029824225964332,0.23470104401044614,0.12619368276848383,0.15000000000000002,0.0,817422.9102820335,504205.2524751482
14,scores_knn1_range4_simc+f.pkl,0.99,0.05765702152069333,0.04038143602483221,0.039256295203914465,0.09333333333333331,0.0,818351.8962202559,491112.75538077945
13,scores_knn1_range4_sima.pkl,0.8,0.6915079954359247,0.8010499900243476,0.650140662950093,0.6233333333333332,0.0,812486.3850775419,513737.0630058826
13,scores_knn1_range4_sima.pkl,0.9,0.42580017231807216,0.5393090272715579,0.35475815634932517,0.38333333333333336,0.0,816637.399150372,506708.95775692543
13,scores_knn1_range4_sima.pkl,0.95,0.29037121778371544,0.37598791571733,0.2551257376338163,0.24,0.0,817436.0503754691,497159.91608901194
13,scores_knn1_range4_sima.pkl,0.975,0.16959555938016327,0.27354394159349155,0.11524273654699822,0.12,0.0,817390.6249447188,501887.1873903782
13,scores_knn1_range4_sima.pkl,0.99,0.043050979661790736,0.026929798504296557,0.038889807147742345,0.06333333333333332,0.0,818367.1817651717,488667.1347082219
10,scores_knn1_range3_simc+f.pkl,0.8,0.6430231942159362,0.7629123794333278,0.572823869881148,0.5933333333333333,0.0,778540.8172304785,455330.8845998898
10,scores_knn1_range3_simc+f.pkl,0.9,0.41973103379512094,0.5978448685595843,0.29134823282577843,0.36999999999999994,0.0,817185.2955229488,505030.0734226224
10,scores_knn1_range3_simc+f.pkl,0.95,0.2778358143087646,0.3665993117726211,0.1969081311536726,0.27,0.0,817242.8064665687,500497.9103151547
10,scores_knn1_range3_simc+f.pkl,0.975,0.1689635852979016,0.2608331266422648,0.10272429591810658,0.1433333333333333,0.0,817478.5680045228,497626.7357071317
10,scores_knn1_range3_simc+f.pkl,0.99,0.03814633545771031,0.041491890164313185,0.02628044954215108,0.04666666666666667,0.0,818587.049870368,486374.47244585824
12,scores_knn1_range4_simc.pkl,0.8,0.6318000058620273,0.7909264744870848,0.49447354309899777,0.6099999999999998,0.0,812312.0833108567,511389.9428638094
12,scores_knn1_range4_simc.pkl,0.9,0.40863678437612416,0.5622094619567543,0.3137008911716182,0.35,0.0,816918.0771113783,505310.7071807583
12,scores_knn1_range4_simc.pkl,0.95,0.26330556858161414,0.3633756740780164,0.20654103166682605,0.21999999999999997,0.0,817310.8087553607,501751.6659022617
12,scores_knn1_range4_simc.pkl,0.975,0.17076265393081133,0.2243638236683646,0.10125747145740266,0.1866666666666667,0.0,817386.1821364816,501738.3450488745
12,scores_knn1_range4_simc.pkl,0.99,0.04616047066497572,0.034760337725124005,0.05372107426980316,0.04999999999999999,0.0,818431.658910607,489738.66045569204
11,scores_knn1_range3_simm.pkl,0.8,0.6321128765706361,0.8150772022934797,0.5179280940850952,0.5633333333333335,0.0,780370.86045217,448576.2652482374
11,scores_knn1_range3_simm.pkl,0.9,0.3769390344057483,0.4704200471083847,0.28373038944219364,0.3766666666666666,0.0,817061.3824462957,499694.23375754524
11,scores_knn1_range3_simm.pkl,0.95,0.24366965966451473,0.33917490712550724,0.15183407186803688,0.2400000000000001,0.0,817582.8405465645,499192.6127571157
11,scores_knn1_range3_simm.pkl,0.975,0.1381163706498611,0.2002256098507912,0.06412350209879215,0.15,0.0,817691.8009551676,490774.30206205306
11,scores_knn1_range3_simm.pkl,0.99,0.04713974765208652,0.03015347708251716,0.03793243254040907,0.07333333333333333,0.0,819482.5359987984,489634.6392181354
8,scores_knn1_range3_simc.pkl,0.8,0.6504944086358677,0.7845976513851513,0.6035522411891185,0.5633333333333332,0.0,779492.2963368967,453566.2028538531
8,scores_knn1_range3_simc.pkl,0.9,0.37241910396878836,0.4563013726164247,0.28095593928994034,0.38000000000000006,0.0,817188.8863608143,500948.65400559735
8,scores_knn1_range3_simc.pkl,0.95,0.21662735674919986,0.2843704800958692,0.19551159015173047,0.16999999999999996,0.0,817222.9777067049,496728.29018799803
8,scores_knn1_range3_simc.pkl,0.975,0.15780538263721813,0.24776264296088105,0.08898683828410661,0.1366666666666667,0.0,817492.6803734085,498826.10165810544
8,scores_knn1_range3_simc.pkl,0.99,0.048729810828674266,0.016231959259021256,0.053290806560334886,0.07666666666666666,0.0,818596.7661072859,488972.90697751427
15,scores_knn1_range4_simm.pkl,0.8,0.6366244249121396,0.7721319987606534,0.4944079426424321,0.6433333333333333,0.0,815295.0419058835,512221.9113133315
15,scores_knn1_range4_simm.pkl,0.9,0.36241871365825684,0.5097240493109783,0.22419875833045896,0.3533333333333333,0.0,817022.3681900754,503943.59078391915
15,scores_knn1_range4_simm.pkl,0.95,0.22956338811937257,0.3891200056094563,0.11290349208199486,0.18666666666666665,0.0,817290.1152770513,503226.3587906514
15,scores_knn1_range4_simm.pkl,0.975,0.1122695042781762,0.1832232732472617,0.036918572920600246,0.11666666666666665,0.0,817817.355350311,497050.03552778874
15,scores_knn1_range4_simm.pkl,0.99,0.0394659728815908,0.024478406213638823,0.027252845764466918,0.06666666666666667,0.0,819253.0823928155,490964.6148692482
9,scores_knn1_range3_sima.pkl,0.8,0.7014194021493452,0.7848535945054408,0.6060712786092611,0.7133333333333334,0.0,778072.6095424662,450162.4154841961
9,scores_knn1_range3_sima.pkl,0.9,0.41114252797135453,0.5424984671449633,0.2775957834357672,0.4133333333333332,0.0,817188.993496297,501166.70386187575
9,scores_knn1_range3_sima.pkl,0.95,0.2560969326570823,0.27724375830757053,0.2110470396636764,0.2799999999999999,0.0,817251.7830329607,496912.88138971414
9,scores_knn1_range3_sima.pkl,0.975,0.09168865046814248,0.10274894502503028,0.03898367304606387,0.1333333333333333,0.0,817488.0749410572,494505.0399410692
9,scores_knn1_range3_sima.pkl,0.99,0.04959161640227103,0.044031141158000366,0.04807704138214605,0.056666666666666664,0.0,818477.1460051072,486298.680390627
4,scores_knn1_range2_simc.pkl,0.8,0.7412319043797999,0.973074369692963,0.6439546767797698,0.6066666666666667,0.0,737596.4730950176,382740.7376369563
4,scores_knn1_range2_simc.pkl,0.9,0.5738094956826967,0.8411389906837461,0.4869561630310107,0.3933333333333333,0.0,783074.9435578217,446100.7112424926
4,scores_knn1_range2_simc.pkl,0.95,0.33428787668399357,0.5418709658249746,0.22765933089367252,0.2333333333333334,0.0,806922.0768836186,494369.63771400356
4,scores_knn1_range2_simc.pkl,0.975,0.09221737373667573,0.07305407133030467,0.07359804987972246,0.13000000000000003,0.0,817319.1963937104,504806.0982629801
4,scores_knn1_range2_simc.pkl,0.99,0.05798383198116557,0.08356939514578197,0.053715434131048054,0.03666666666666667,0.0,817586.7123031208,498008.789169836
3,scores_knn1_range1_simm.pkl,0.8,0.720854559930488,1.0,0.5625636797914639,0.6,0.0,705422.416626279,359390.0635641641
3,scores_knn1_range1_simm.pkl,0.9,0.5422917729602148,0.8369033183673212,0.40330533384665673,0.38666666666666666,0.0,778200.4419515243,421806.00881097926
3,scores_knn1_range1_simm.pkl,0.95,0.38469217016041485,0.6115004938087361,0.209242683339175,0.33333333333333337,0.0,804343.5388118275,482472.1932627537
3,scores_knn1_range1_simm.pkl,0.975,0.10452286123624183,0.12786599279495278,0.03570259091377266,0.15000000000000002,0.0,817354.5224496073,503357.8649277356
3,scores_knn1_range1_simm.pkl,0.99,0.06104671035571686,0.08041466498329784,0.03272546608385277,0.06999999999999998,0.0,818483.6886208794,498890.44748947414
7,scores_knn1_range2_simm.pkl,0.8,0.7448513379241085,0.9584312438827971,0.6461227698895287,0.6299999999999998,0.0,749571.4713202502,386201.4174999823
7,scores_knn1_range2_simm.pkl,0.9,0.5462573776048762,0.828533215389463,0.38023891742516575,0.42999999999999994,0.0,781292.6811680082,447306.26475688047
7,scores_knn1_range2_simm.pkl,0.95,0.3217427201403903,0.5787977943157047,0.1664303661054663,0.21999999999999997,0.0,804700.1070413872,473205.7279182959
7,scores_knn1_range2_simm.pkl,0.975,0.08158217218113305,0.09725312061908076,0.03416006259098504,0.11333333333333334,0.0,817994.1769402333,504545.3222151689
7,scores_knn1_range2_simm.pkl,0.99,0.06662288099600966,0.08413956643756583,0.04239574321712984,0.0733333333333333,0.0,818568.690097454,498436.1458881873
1,scores_knn1_range1_sima.pkl,0.8,0.780241146545791,0.98,0.6940567729707063,0.6666666666666665,0.0,714577.301116427,366912.025166415
1,scores_knn1_range1_sima.pkl,0.9,0.5764836538957608,0.8406215464580491,0.4388294152292334,0.44999999999999996,0.0,772633.0879714403,428378.1740903585
1,scores_knn1_range1_sima.pkl,0.95,0.37389742539946935,0.6247840031324912,0.2702416063992503,0.2266666666666667,0.0,809612.6130510849,493972.25792530447
1,scores_knn1_range1_sima.pkl,0.975,0.08127683007875645,0.07101508086151626,0.0661487427080864,0.10666666666666667,0.0,817258.8724348969,503276.4083177677
1,scores_knn1_range1_sima.pkl,0.99,0.06838137830502564,0.10074690267871647,0.04439723223636047,0.059999999999999984,0.0,817422.751667468,500214.2353782655
0,scores_knn1_range1_simc.pkl,0.8,0.7560426225430035,0.99,0.7014612009623444,0.5766666666666663,0.0,716172.0193111924,366023.95207132655
0,scores_knn1_range1_simc.pkl,0.9,0.5675204220865578,0.8199417716364951,0.4626194946231786,0.42,0.0,773459.0999157719,433774.8980121353
0,scores_knn1_range1_simc.pkl,0.95,0.3687033835636099,0.5520641514555781,0.2740459992352517,0.27999999999999986,0.0,809202.083734669,495531.3444576461
0,scores_knn1_range1_simc.pkl,0.975,0.09629909583609099,0.11043156009512695,0.07179906074647933,0.1066666666666667,0.0,817257.1133315187,508733.6860007684
0,scores_knn1_range1_simc.pkl,0.99,0.05530014804655714,0.0573041610682729,0.06192961640473188,0.04666666666666666,0.0,817561.8216412914,503271.77027880354
2,scores_knn1_range1_simc+f.pkl,0.8,0.7450024608597007,0.98,0.6250073825791018,0.6300000000000002,0.0,712574.8061346279,363470.6268829829
2,scores_knn1_range1_simc+f.pkl,0.9,0.5974483505087094,0.8165938217849407,0.4957512297411874,0.47999999999999976,0.0,772505.2400169709,434278.46632506244
2,scores_knn1_range1_simc+f.pkl,0.95,0.3688104207021356,0.5881132044183546,0.2783180576880521,0.23999999999999994,0.0,809215.775317097,499394.74419924174
2,scores_knn1_range1_simc+f.pkl,0.975,0.10936957829349926,0.0851776121509855,0.09293112272951226,0.15,0.0,817232.4335396413,504308.17438670713
2,scores_knn1_range1_simc+f.pkl,0.99,0.07393678231097388,0.09720340952692999,0.05127360407265834,0.07333333333333333,0.0,817491.9936535408,504628.89890537265
5,scores_knn1_range2_sima.pkl,0.8,0.7596551039045704,0.9640349620561836,0.6615970163241947,0.6533333333333333,0.0,735878.3133083908,379388.4532229259
5,scores_knn1_range2_sima.pkl,0.9,0.508380991571301,0.7191880723394128,0.37595490237449025,0.4299999999999999,0.0,782670.2763111828,440130.25019348453
5,scores_knn1_range2_sima.pkl,0.95,0.3691422788790458,0.6277430481013423,0.26301712186912857,0.21666666666666662,0.0,808449.3313051553,493472.62433754065
5,scores_knn1_range2_sima.pkl,0.975,0.212693701425082,0.34990029818755597,0.18151413942102348,0.10666666666666666,0.0,817263.2380352368,504353.2770155734
5,scores_knn1_range2_sima.pkl,0.99,0.054979544532061435,0.05986754630644146,0.05173775395640951,0.05333333333333333,0.0,817614.6068285914,497580.11603664036
6,scores_knn1_range2_simc+f.pkl,0.8,0.7617202472827408,0.9717452516762699,0.6667488235052862,0.6466666666666664,0.0,733120.2503641521,378432.81713666755
6,scores_knn1_range2_simc+f.pkl,0.9,0.5723915376848001,0.8061374522617899,0.47437049412594356,0.43666666666666676,0.0,783523.0412515849,443737.82518039766
6,scores_knn1_range2_simc+f.pkl,0.95,0.3484599415530943,0.5890499239107694,0.23632990074851323,0.22000000000000008,0.0,808384.1923137064,499211.0237901659
6,scores_knn1_range2_simc+f.pkl,0.975,0.10293439317835809,0.07989047785701557,0.072246035011392,0.1566666666666667,0.0,817225.4574195032,502688.9363724643
6,scores_knn1_range2_simc+f.pkl,0.99,0.06649793727086933,0.08602560150118801,0.060134876978086664,0.05333333333333333,0.0,817564.0842133972,501991.2148732553"""

if __name__ == '__main__':
    plot_experimental_results(data, output_filename="experimental_results_plot_2.pdf", core_f1_only=True)