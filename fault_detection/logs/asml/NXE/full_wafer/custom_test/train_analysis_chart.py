import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from pathlib import Path
import os
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Set IEEE font globally
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['text.usetex'] = False
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{mathrsfs}'

def plot_eval_chart(excel_path, model_name):
    df = pd.read_excel(excel_path)
    num_traj_values = sorted(df['num_traj'].unique())
    bg_colors = {'recall': "#8FB9F3CB", 'precision': "#BB90D0B1"}
    box_colors = {'recall': "#6C8EBF", 'precision': "#9673A6"}
    font_colors = {'recall': "#4C6BAF", 'precision': "#6A5093"}
    min_width = 0.0

    def draw_box(ax, type, x, y, width, height, box_color, bg_color, edge=None):
        hatch_pattern = '\\' if type == "precision" else '/'
        if type == "precision":
            bg_rect = patches.Rectangle((0, 0), -1, 1, color=bg_color, alpha=0.3, hatch=hatch_pattern)
        elif type == "recall":
            bg_rect = patches.Rectangle((0, 0), 1, 1, color=bg_color, alpha=0.3, hatch=hatch_pattern)
        rect = patches.Rectangle((x, y), width, height, color=box_color, alpha=1)
        ax.add_patch(bg_rect)
        ax.add_patch(rect)
        linewidth = 8
        if edge == 'left':
            ax.plot([x, x], [y, y + height], color=box_colors['precision'], linestyle='-', linewidth=linewidth)
            ax.plot([x, x + width], [y + height, y + height], color=box_colors['precision'], linestyle='-', linewidth=linewidth)
        elif edge == 'right':
            ax.plot([x + width, x + width], [y, y + height], color=box_colors['recall'], linestyle='-', linewidth=linewidth)
            ax.plot([x, x + width], [y + height, y + height], color=box_colors['recall'], linestyle='-', linewidth=linewidth)

    fig, axes = plt.subplots(1, len(num_traj_values), figsize=(8 * len(num_traj_values), 11), sharey=False)
    if len(num_traj_values) == 1:
        axes = [axes]
    fig.patch.set_alpha(0)

    main_fontsize = 65
    for i, num_traj in enumerate(num_traj_values):
        ax = axes[i]
        ax.patch.set_alpha(0.0)
        data = df[df['num_traj'] == num_traj].iloc[0]
        zero_pos = 0
        bar_start = 0
        bar_width = data['cr'] + zero_pos - bar_start

        # Precision box
        prec_height = data['precision']
        prec_width = max(bar_width, min_width)
        prec_x = -bar_start - prec_width
        draw_box(ax, "precision", prec_x, 0, prec_width, prec_height, box_colors['precision'], bg_colors['precision'], edge=None)
        ax.text(
            prec_x + prec_width / 2, prec_height + 0.01,
            f"{prec_height:.2f}",
            ha='center', va='bottom',
            color=font_colors['precision'],
            fontsize=main_fontsize,
            fontweight='bold'
        )

        # Recall box
        recall_height = data['recall']
        recall_width = max(bar_width, min_width)
        recall_x = bar_start
        draw_box(ax, "recall", recall_x, 0, recall_width, recall_height, box_colors['recall'], bg_colors['recall'], edge=None)
        ax.text(
            recall_x + recall_width / 2, recall_height + 0.01,
            f"{recall_height:.2f}",
            ha='center', va='bottom',
            color=font_colors['recall'],
            fontsize=main_fontsize,
            fontweight='bold'
        )

        ax.axvline(0, color='black', linestyle='dotted', linewidth=6, alpha=0.7)

        # Custom x-axis ticks: [1, 0.5, 0, 0.5, 1]
        xticks = [-zero_pos-(2*0.5), -zero_pos-0.5, 0, zero_pos+(1*0.5), zero_pos+(2*0.5)]
        ax.set_xticks(xticks)
        xticklabels = ['1', '0.5', '0', '0.5', '1']
        ax.set_xticklabels(xticklabels, fontsize=main_fontsize)

        # Set all x-ticks to default length
        ax.tick_params(axis='x', length=20, width=4)

        # Color x-tick labels
        for idx, label in enumerate(ax.get_xticklabels()):
            if idx <= 1:
                label.set_color(font_colors['precision'])
            elif idx >= 3:
                label.set_color(font_colors['recall'])
            else:
                label.set_color('black')

        ax.set_xlim(-zero_pos-(2*0.5), zero_pos+(2*0.5))
        ax.set_xlabel(r"$C^{\text{(TP)}}_{r}$", fontsize=main_fontsize)
        if i == 0:
            ax.set_ylabel(r"$P_r \ \text{or} \ \ R_e$", fontsize=main_fontsize*1.2)

        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=main_fontsize)
        ax.set_ylim(0, 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_visible(False) if i != 0 else ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['bottom'].set_color('black')

        ax.tick_params(axis='y', length=20, width=4)

        if i != 0:
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.set_ylabel("")

    plt.tight_layout(rect=[0, 0.18, 1, 1])  # leave space at bottom for arrow axis
    plt.subplots_adjust(wspace=0.5)

    # Draw custom "axis" with arrow and num_traj values below all subplots
    arrow_y = 0.000  # fraction below axes
    arrow_start = 0.08
    arrow_end = 1
    arrow = FancyArrowPatch(
        (arrow_start, arrow_y), (arrow_end, arrow_y),
        transform=fig.transFigure,
        arrowstyle='->', color='black', linewidth=6, mutation_scale=120
    )
    fig.add_artist(arrow)

    # Annotate each num_traj value above the arrow in LaTeX as M = <traj value>
    for idx, (ax, val) in enumerate(zip(axes, num_traj_values)):
        ax_pos = ax.get_position()
        center_x = (ax_pos.x0 + ax_pos.x1) / 2
        latex_str = rf"$M = {val}$"
        fig.text(center_x, arrow_y + 0.05, latex_str, ha='center', va='bottom', fontsize=main_fontsize*1.3, fontweight='bold')

    # Place axis label below the arrow
    fig.text(0.5, arrow_y - 0.05, r"$\text{Number of train trajectories } (M)$", ha='center', va='top', fontsize=main_fontsize * 1.3, fontweight='normal')

    # Draw vertical lines between subplots
    for i in range(1, len(axes)):
        left_ax = axes[i-1]
        right_ax = axes[i]
        # Get the right edge of the left subplot and left edge of the right subplot
        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()
        # Compute the x position between subplots
        x = (left_pos.x1 + right_pos.x0) / 2
        # Draw vertical line from top to bottom of figure
        fig.lines.append(plt.Line2D([x, x], [0.05, 0.97], color='grey', linewidth=5, linestyle='--', transform=fig.transFigure))

    # Save fig
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, '_train_analysis_plots', f'{model_name}_train_data_eval.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    plot_specs = [
        {'model_name': 'ammf_acc', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\custom_test\ammf\acc\set_E1\train_data_eval.xlsx")},
        #{'model_name': 'ammf_ctrl', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\custom_test\ammf\ctrl\set_E1\iswp_1\eval.xlsx")},

        #{'model_name': 'pob_los_ctrl', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\custom_test\pob_los\ctrl\set_E1\iswp_1\eval.xlsx")},
        
        {'model_name': 'ws_ls1_pos', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\custom_test\ws_ls1\pos\set_E1\train_data_eval.xlsx")},
        #{'model_name': 'ws_ls1_ctrl', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\asml\NXE\full_wafer\custom_test\ws_ls1\ctrl\set_E1\iswp_1\eval.xlsx")},
    ]
    for spec in plot_specs:
        excel_path = spec['excel_path']
        model_name = spec['model_name']
        plot_eval_chart(excel_path, model_name)