import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from pathlib import Path
import os

# Set IEEE font globally
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['text.usetex'] = False
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{mathrsfs}'

def plot_eval_chart(excel_path, model_name):
    df = pd.read_excel(excel_path)
    features = ['f_prop', 'f_bench']
    bg_colors = {'recall': "#8FB9F3CB", 'precision': "#BB90D0B1"}  # purple, blue from same palette
    box_colors = {'recall': "#6C8EBF", 'precision': "#9673A6"}  # darker purple, darker blue
    font_colors = {'recall': "#4C6BAF", 'precision': "#6A5093"}  # even darker purple, darker blue
    min_width = 0.0  # minimum box width

    def draw_box(ax, type, x, y, width, height, box_color, bg_color, edge=None):
        # if type == "precision":
        #     bg_rect = patches.Rectangle((0, 0), -1, 1, color=bg_color, alpha=1)
        # elif type == "recall":
        #     bg_rect = patches.Rectangle((0, 0), 1, 1, color=bg_color, alpha=1)
        hatch_pattern = '\\' if type == "precision" else '/'  # example: diagonal lines
        if type == "precision":
            bg_rect = patches.Rectangle((0, 0), -1, 1, color=bg_color, alpha=0.3, hatch=hatch_pattern)
        elif type == "recall":
            bg_rect = patches.Rectangle((0, 0), 1, 1, color=bg_color, alpha=0.3, hatch=hatch_pattern)

        rect = patches.Rectangle((x, y), width, height, color=box_color, alpha=1)   
        ax.add_patch(bg_rect)
        ax.add_patch(rect)
        linewidth = 8
        # Draw dotted edges using ax.plot
        if edge == 'left':
            # Left edge
            ax.plot([x, x], [y, y + height], color=box_colors['precision'], linestyle='-', linewidth=linewidth)
            # Top edge
            ax.plot([x, x + width], [y + height, y + height], color=box_colors['precision'], linestyle='-', linewidth=linewidth)
        elif edge == 'right':
            # Right edge
            ax.plot([x + width, x + width], [y, y + height], color=box_colors['recall'], linestyle='-', linewidth=linewidth)
            # Top edge
            ax.plot([x, x + width], [y + height, y + height], color=box_colors['recall'], linestyle='-', linewidth=linewidth)

    fig, axes = plt.subplots(1, 2, figsize=(22, 11), sharey=True)
    fig.patch.set_alpha(0.0)
    
    for i, feature in enumerate(features):
        ax = axes[i]
        if i == 0:
            ax.patch.set_facecolor("#F0E80A")
            ax.patch.set_alpha(0.0)
        else:
            ax.patch.set_alpha(0.0)  # fully transparent
        data = df[df['feature'] == feature].iloc[0]
        zero_pos = 0
        bar_start = 0
        bar_width = data['cr']+zero_pos-bar_start
        main_fontsize = 55

        # Precision box (extends leftwards)
        prec_height = data['precision']
        prec_width = max(bar_width, min_width)
        prec_x = -bar_start - prec_width  # left edge
        draw_box(ax, "precision", prec_x, 0, prec_width, prec_height, box_colors['precision'], bg_colors['precision'], edge=None)
        # Annotate precision value at top center
        ax.text(
            prec_x + prec_width / 2, prec_height + 0.01,
            f"{prec_height:.2f}",
            ha='center', va='bottom',
            color=font_colors['precision'],
            fontsize=main_fontsize,
            fontweight='bold'
        )

        # Recall box (extends rightwards)
        recall_height = data['recall']
        recall_width = max(bar_width, min_width)
        recall_x = bar_start  # left edge
        draw_box(ax, "recall", recall_x, 0, recall_width, recall_height, box_colors['recall'], bg_colors['recall'], edge=None)
        
        # Annotate recall value at top center
        ax.text(
            recall_x + recall_width / 2, recall_height + 0.01,
            f"{recall_height:.2f}",
            ha='center', va='bottom',
            color=font_colors['recall'], 
            fontsize=main_fontsize,
            fontweight='bold'
        )
        
        # Dotted vertical line
        ax.axvline(0, color='black', linestyle='dotted', linewidth=6, alpha=0.7)
        # ax.axvline(zero_pos, color='black', linestyle='dotted', linewidth=8, alpha=0.7)

        # Custom x-axis ticks: [1, 0.5, 0, 0, 0.5, 1]
        ax.set_xticks([-zero_pos-(2*0.5), -zero_pos-0.5, 0, zero_pos+(1*0.5), zero_pos+(2*0.5)])
        ax.set_xticklabels(['1', '0.5', '0', '0.5', '1'], fontsize=main_fontsize)

        # Color x-tick labels under each box
        xtick_labels = ax.get_xticklabels()
        for idx, label in enumerate(xtick_labels):
            if idx <= 1:  # left three ticks (precision)
                label.set_color(font_colors['precision'])
            elif idx >= 3:  # right three ticks (recall)
                label.set_color(font_colors['recall'])
            else:  # middle tick (empty)
                label.set_color('grey')

        ax.set_xlim(-zero_pos-(2*0.5)-0.15, zero_pos+(2*0.5)+0.3)
        ax.set_xlabel(r"$C^{\text{(TP)}}_{r}$", fontsize=main_fontsize*1)
        if i == 0:
            ax.set_ylabel(r"$P_r \ \text{or} \ \ R_e$", fontsize=main_fontsize*1)

        # Place arrow to the left of x-label
        arrow_pos = -0.28
        ax.annotate(
            '',
            xy=(0.1, arrow_pos),  # start near left
            xytext=(0.38, arrow_pos),  # end near center-left
            xycoords='axes fraction',
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='grey', linewidth=5, alpha=1, mutation_scale=80)
        )

        # Place arrow to the right of x-label
        ax.annotate(
            '',
            xy=(0.62, arrow_pos),  # start near center-right
            xytext=(0.9, arrow_pos),  # end near right
            xycoords='axes fraction',
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle='<-', color='grey', linewidth=5, alpha=1, mutation_scale=80)
        )

        # Add feature name in LaTeX below x-axis label
        if '_' in feature:
            main, sub = feature.split('_', 1)
            main = r"\mathscr{M}"
            if sub == "bench":
                sub = "baseline"
            latex_feat = f"${main}_{{\\text{{{sub}}}}}$"
        else:
            latex_feat = f"${feature}$"
        ax.text(0.5, -0.4, latex_feat, transform=ax.transAxes,
                ha='center', va='top', fontsize=main_fontsize*1.4)
        
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=main_fontsize)
        ax.set_ylim(0, 1.2)
        # Draw box around subplot
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_visible(False) if i == 1 else ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['bottom'].set_color('black')

        ax.tick_params(axis='x', length=20, width=4)  # Increase x-tick length and width
        ax.tick_params(axis='y', length=20, width=4)  # Increase y-tick length and width (optional)
        # for spine in ax.spines.values():
        #     spine.set_visible(True)
            # spine.set_linewidth(1)
            # spine.set_color('grey')

    plt.tight_layout()

    # Draw vertical line between the two subplots
    fig_width = fig.get_figwidth()
    fig_height = fig.get_figheight()
    # Get the axes positions in figure coordinates
    ax1_pos = axes[0].get_position()
    ax2_pos = axes[1].get_position()
    # x position between the two axes
    x_between = (ax1_pos.x1 + ax2_pos.x0) / 2
    # Draw the line from top to bottom of the figure
    fig.lines.append(plt.Line2D([x_between, x_between], [0.09, 1], transform=fig.transFigure, color='grey', linewidth=8, alpha=0.7))

    # save fig
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, '_plots', f'{model_name}_eval.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    plot_specs = [
        {'model_name': 'ammf_acc', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\custom_test\ammf\acc\set_E1\iswp_1\eval.xlsx")},
        {'model_name': 'ammf_ctrl', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\custom_test\ammf\ctrl\set_E1\iswp_1\eval.xlsx")},

        {'model_name': 'pob_los_ctrl', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\custom_test\pob_los\ctrl\set_E1\iswp_1\eval.xlsx")},
        
        {'model_name': 'ws_ls1_pos', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\custom_test\ws_ls1\pos\set_E1\iswp_1\eval.xlsx")},
        {'model_name': 'ws_ls1_ctrl', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\custom_test\ws_ls1\ctrl\set_E1\iswp_1\eval.xlsx")},
    ]
    for spec in plot_specs:
        excel_path = spec['excel_path']
        model_name = spec['model_name']
        plot_eval_chart(excel_path, model_name)