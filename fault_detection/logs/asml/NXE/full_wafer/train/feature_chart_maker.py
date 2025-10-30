import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import numpy as np
import os
from matplotlib.patches import FancyBboxPatch

# Set IEEE font globally
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def plot_feature_importance_from_excel(
    excel_path,
    theme='red',
    bg_alpha=0.0
):
    # More varied theme color palettes
    themes = {
        'red': {
            'bars': ['#e74c3c', "#d63031", '#d63031', '#c0392b', '#fd5c63'],
            'edge': '#b71c1c',
            'label': '#6e2c00',
            'bg': "#ffe4e4"
        },
        'green': {
            'bars': ['#27ae60', '#177f44', "#177f44", '#00cec9', '#b2ff59'],
            'edge': '#145a32',
            'label': '#004d40',
            'bg': "#e7f8e4"
        },
        'light_green': {
            'bars': ['#27ae60', "#7ccd1a", "#7ccd1a", '#00cec9', '#b2ff59'],
            'edge': '#145a32',
            'label': "#1f4d00",
            'bg': "#e7f8e4"
        },
        'yellow': {
            'bars': ['#f7dc6f', '#ffe082', '#fff176', '#fbc02d', '#ffd600'],
            'edge': '#b7950b',
            'label': '#7f6000',
            'bg': "#fff8e4"
        },
        'orange': {
            'bars': ['#e67e22', "#ff7043", '#ff7043', '#ca6f1e', '#ff9800'],
            'edge': '#af601a',
            'label': '#6e2c00',
            'bg': "#f2dcca"
        }
    }
    colors = themes.get(theme, themes['red'])

    # Read Excel
    df = pd.read_excel(excel_path)

    # Prepare subplot 1 data
    x1 = df['short form time'].astype(str)
    y1 = df['normalized scores']

    # Prepare subplot 2 data
    x2 = df['short form freq'].astype(str)
    y2 = df['freq scores']

    # Format x1: 'ft1' -> 'f_{t_{1}}'
    def format_subscript(label):
        if label.startswith('ft'):
            num = label[2:]
            return f"$f_{{t_{{{num}}}}}$"
        elif label.startswith('f'):
            return f"$f_{{{label[1:]}}}$"
        return label

    x1_labels = [format_subscript(lbl) for lbl in x1]

    # Format x2: 'fw1' -> 'f_{ω_{1}}'
    def format_freq(label):
        if label.startswith('fw'):
            num = label[2:]
            return f"$f_{{\\omega_{{{num}}}}}$"
        elif label.startswith('f'):
            return f"$f_{{{label[1:]}}}$"
        return label
    
    def latex_set(indices, var, max_per_row=4):
        indices = [str(i+1) for i in indices]
        rows = [indices[i:i+max_per_row] for i in range(0, len(indices), max_per_row)]
        latex_rows = []
        for idx, row in enumerate(rows):
            if idx == 0:
                latex_rows.append(fr"{var} \mid &\ i = {', '.join(row)} \\")
            else:
                latex_rows.append(fr"\mid &\ i = {', '.join(row)} \\")
        latex_body = ''.join(latex_rows).rstrip('\\')
        latex = (
            r"\left\{ \begin{aligned}"
            f"{latex_body}"
            r"\end{aligned} \right\}"
        )
        return latex

    x2_labels = [format_freq(lbl) for lbl in x2]

    # Set bar width and x locations for both subplots
    bar_width = 0.9

    # Calculate spacing for x-ticks to avoid congestion
    def get_tick_indices(n, max_ticks=7):
        if n <= max_ticks:
            return np.arange(n)
        step = int(np.ceil(n / max_ticks))
        return np.arange(0, n, step)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 10.5), sharey=True)
    fig.patch.set_facecolor(colors['bg'])
    fig.patch.set_alpha(bg_alpha)
    main_fontsize = 110
    reduce_factor = 0.45

    # Subplot 1 (37 values)
    n1 = len(x1_labels)
    # bar_colors1 = (colors['bars'] * ((n1 // len(colors['bars'])) + 1))[:n1]
    bar_colors1 = colors['bars'][2]
    bars1 = axes[0].bar(
        np.arange(n1), y1, width=bar_width,
        color=bar_colors1, edgecolor=colors['edge']
    )
    #axes[0].set_xlabel("Time features", color=colors['label'], fontsize=main_fontsize*0.65)
    axes[0].set_ylabel("Importance", color=colors['label'], fontsize=main_fontsize*0.6)
    axes[0].set_facecolor(colors['bg'])
    axes[0].patch.set_alpha(bg_alpha)

    # X-ticks for all values, but only label every 5th
    tick_idx1 = np.arange(n1)
    tick_labels1 = [x1_labels[i] if i % 2 == 0 else "" for i in tick_idx1]
    axes[0].set_xticks(tick_idx1)
    #axes[0].set_xticklabels(tick_labels1, rotation=0, ha='center', color=colors['label'], fontsize=main_fontsize, fontname='Times New Roman')
    axes[0].set_xticklabels([''] * n1)

    # Make every 5th tick a major tick (longer), others minor (shorter)
    axes[0].set_xticks(tick_idx1, minor=True)
    major_ticks = tick_idx1[::5]
    axes[0].set_xticks(major_ticks, minor=False)
    axes[0].tick_params(axis='x', which='major', length=35, width=2)
    axes[0].tick_params(axis='x', which='minor', length=28, width=1.5)

    axes[0].tick_params(axis='y', colors=colors['label'], labelsize=main_fontsize)
    

    #  # Annotate only non-zero bars in subplot 1, dynamically adjust position if overlap
    # prev_x = None
    # prev_height = None
    # annotation_fontsize = main_fontsize * reduce_factor  # Large font

    # for idx, (rect, label) in enumerate(zip(bars1, x1_labels)):
    #     height = rect.get_height()
    #     if height != 0:
    #         x = rect.get_x() + rect.get_width() / 2
    #         # Default offset
    #         offset = 10
    #         # If previous bar is close and heights are similar, use higher offset
    #         if prev_x is not None and abs(x - prev_x) < 3 * rect.get_width():
    #             if prev_height is not None and abs(height - prev_height) < 0.1 * y1.max():
    #                 offset = 50  # Higher offset if heights are similar and bars are close
    #         axes[0].annotate(
    #             label,
    #             xy=(x, height),
    #             xytext=(0, offset),
    #             textcoords="offset points",
    #             ha='center', va='bottom', color=colors['edge'], fontsize=annotation_fontsize, fontname='Times New Roman'
    #         )
    #         prev_x = x
    #         prev_height = height

    # Add feature set below plot 1

    nonzero_idx1 = [i for i, v in enumerate(y1) if v != 0]
    set_str1 = latex_set(nonzero_idx1, "f_{t(i)}")

    axes[0].text(
        0.5, -0.28,  # Fixed position below x-axis
        f"${set_str1}$",
        fontsize=main_fontsize*0.7,
        color=colors['label'],
        ha='center', va='top',
        transform=axes[0].transAxes,
        clip_on=False
    )

    arrow_pos = -0.23
    axes[0].annotate(
        '',
        xy=(0.25, arrow_pos),  # start near left
        xytext=(0.75, arrow_pos),  # end near center-left
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(arrowstyle='<-', color=colors['label'], linewidth=6, alpha=1, mutation_scale=80)
    )


    # Subplot 2 (10 values)
    n2 = 10
    x2_labels = x2_labels[:n2]
    y2 = y2[:n2]
    bar_colors2 = colors['bars'][1]
    x2_pos = np.linspace(0, n1-1, n2)  # Spread bars across same axis length as subplot 1

    bars2 = axes[1].bar(
        x2_pos, y2, width=bar_width,
        color=bar_colors2, edgecolor=colors['edge']
    )
    #axes[1].set_xlabel("Frequency features", color=colors['label'], fontsize=main_fontsize*0.65)
    axes[1].set_facecolor(colors['bg'])
    axes[1].patch.set_alpha(bg_alpha)

    # Major ticks at even positions (0,2,4,6,8)
    major_ticks = x2_pos[::2]
    axes[1].set_xticks(major_ticks)
    #axes[1].set_xticklabels([x2_labels[i] for i in range(0, n2, 2)], rotation=0, ha='center', color=colors['label'], fontsize=main_fontsize)
    axes[1].set_xticklabels([''] * major_ticks.shape[0])

    # Minor ticks at odd positions (1,3,5,7,9)
    minor_ticks = x2_pos[1::2]
    axes[1].set_xticks(minor_ticks, minor=True)

    axes[1].tick_params(axis='x', which='major', length=35, width=2)
    axes[1].tick_params(axis='x', which='minor', length=35, width=2)
    axes[1].tick_params(axis='y', colors=colors['label'])

    # # Annotate only non-zero bars in subplot 2
    # for idx, (rect, label) in enumerate(zip(bars2, x2_labels)):
    #     height = rect.get_height()
    #     if height != 0:
    #         axes[1].annotate(
    #             label,
    #             xy=(rect.get_x() + rect.get_width() / 2, height),
    #             xytext=(0, 3),
    #             textcoords="offset points",
    #             ha='center', va='bottom', color=colors['edge'], fontsize=main_fontsize * reduce_factor
    #         )

    # Add feature set below plot 2
    nonzero_idx2 = [i for i, v in enumerate(y2) if v != 0]
    set_str2 = latex_set(nonzero_idx2, "\mathbf{f}_{\omega(i)}", max_per_row=3)

    axes[1].text(
        0.5, -0.28,  # Fixed position below x-axis
        f"${set_str2}$",
        fontsize=main_fontsize*0.7,
        color=colors['label'],
        ha='center', va='top',
        transform=axes[1].transAxes,
        clip_on=False
    )

    arrow_pos = -0.23
    axes[1].annotate(
        '',
        xy=(0.25, arrow_pos),  # start near left
        xytext=(0.75, arrow_pos),  # end near center-left
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(arrowstyle='<-', color=colors['label'], linewidth=6, alpha=1, mutation_scale=80)
    )

    # Set same y-limits for both subplots for visual consistency
    y_max = max(max(y1), max(y2)) * 1.1
    axes[0].set_ylim(0, y_max)
    axes[1].set_ylim(0, y_max)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # Set to desired thickness
            spine.set_color(colors['label'])  

    plt.tight_layout()  # Increase pad value for more space at top
    plt.subplots_adjust(top=0.99, left=0.09, right=0.98, bottom=0.62) 
    # save fig
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.basename(os.path.dirname(excel_path))
    save_path = os.path.join(script_dir, 'plots', f'{model_name}_feats.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    plot_specs = [
        # ammf - acc
        {'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\train\ammf\acc\set_E1\IF\tswp_1\[ammf_(acc+E1)]-IF_fdet_1\feature_sheet.xlsx"),
        'theme': 'green'},
        # ammf - ctrl
        {'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\train\ammf\ctrl\set_E1\IF\tswp_1\[ammf_(ctrl+E1)]-IF_fdet_1\feature_sheet.xlsx"),
        'theme': 'green'},

        # pob_los - ctrl
        {'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\train\pob_los\ctrl\set_E1\IF\tswp_1\[pob_los_(ctrl+E1)]-IF_fdet_1\feature_sheet.xlsx"),
        'theme': 'green'},

        # ws_ls1 - ctrl
        {'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\train\ws_ls1\ctrl\set_E1\IF\tswp_1\[ws_ls1_(ctrl+E1)]-IF_fdet_3\feature_sheet.xlsx"),
        'theme': 'green'},
        # ws_ls1 - pos
        {'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\train\ws_ls1\pos\set_E1\IF\tswp_1\[ws_ls1_(pos+E1)]-IF_fdet_1\feature_sheet.xlsx"),
        'theme': 'light_green'},
    ]


    for spec in plot_specs:
        plot_feature_importance_from_excel(spec['excel_path'], theme=spec['theme'])

 