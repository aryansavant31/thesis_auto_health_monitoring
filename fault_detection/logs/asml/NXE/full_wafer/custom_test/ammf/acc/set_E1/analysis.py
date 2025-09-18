import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.patches as mpatches


def IF_histogram_plot(excel_path, fault_level, precison_weight=0.5):

    df = pd.read_excel(excel_path, sheet_name=0)
    book_name = os.path.basename(excel_path).replace('.xlsx', '')

    recall_weight = 1 - precison_weight

    # filter out fault level
    if fault_level == 'subtle':
        fault_level_col = 'custom_test_3'
        fault_level_caption = 'Low Severity Faults'
    elif fault_level == 'obvious':
        fault_level_col = 'custom_test_1'
        fault_level_caption = 'High Severity Faults'
    elif fault_level == 'medium':
        fault_level_col = 'custom_test_2'
        fault_level_caption = 'Medium Severity Faults'

    filtered_df = df[df['run_type'] == fault_level_col]

    # Create a pair label for each row
    def make_label(row):
        n_est = int(row['IF/n_estimators'])
        contam = row['IF/contam']
        contam_str = f"{contam:.0e}" if contam < 1e-2 else f"{contam:.2f}"
        return f"({n_est}, {contam_str})"
    filtered_df['pair_label'] = filtered_df.apply(make_label, axis=1)

    # Sort by contam then n_est
    filtered_df = filtered_df.sort_values(['IF/contam', 'IF/n_estimators'])

    new_rows = []
    j = 1
    new_rows.append({'pair_label': f'Group {j}', 'precision': np.nan, 'recall': np.nan})
    prev_contam = None
    for idx, row in filtered_df.iterrows():
        contam = row['IF/contam']
        if prev_contam is not None and contam != prev_contam:
            j += 1
            # Insert an empty row for separation
            new_rows.append({'pair_label': f'Group {j}', 'precision': np.nan, 'recall': np.nan})
        new_rows.append(row.to_dict())  # <-- Convert row to dict
        prev_contam = contam

    plot_df = pd.DataFrame(new_rows)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    # Find best overall metric pair (highest mean of precision and recall)
    valid_bins = plot_df[~plot_df['pair_label'].str.startswith('Group ')]
    overall_means = valid_bins.groupby('pair_label').apply(
        lambda g: precison_weight * g['precision'].mean() + recall_weight * g['recall'].mean()
    )
    best_pair = overall_means.idxmax()

    # Precision variation plot
    fig, ax = plt.subplots(figsize=(15, 12), dpi=100)
    sns.boxplot(y='pair_label', x='precision', data=plot_df, ax=ax, color='skyblue', showfliers=False)
    sns.stripplot(y='pair_label', x='precision', data=plot_df, ax=ax, color='navy', size=6, jitter=True)
    ax.set_title(f'Precision Variation for Isolation Forest Hyperparameters (precision weight = {precison_weight}) : [{fault_level_caption}]', fontsize=11, pad=15)
    ax.set_xlabel('Precision')
    ax.set_ylabel('(n_trees, contam)')
    
    plt.grid(axis='y')

    for i, label in enumerate(ax.get_yticklabels()):
        if label.get_text().startswith('Group '):
            ax.axhspan(i-0.05, i+0.05, color='black', alpha=0.45)
        if label.get_text() == best_pair:
            ax.axhspan(i-0.3, i+0.3, color='darkgreen', alpha=0.15, zorder=0)

    # Highlight min and max bins
    grouped = plot_df.groupby('pair_label')['precision']
    mean_precisions = grouped.mean()
    min_idx = mean_precisions.idxmin()
    max_idx = mean_precisions.idxmax()
    yticks = ax.get_yticklabels()
    for i, label in enumerate(yticks):
        if label.get_text() == min_idx:
            ax.axhspan(i-0.3, i+0.3, color='darkred', alpha=0.15)
        if label.get_text() == max_idx:
            ax.axhspan(i-0.3, i+0.3, color='darkblue', alpha=0.15)

    handles = [
        mpatches.Patch(color='orange', alpha=0.15, label='h.p. (n_trees, contam)'),
        mpatches.Patch(color='darkgreen', alpha=0.15, label='Best Pair'),
        mpatches.Patch(color='darkred', alpha=0.15, label='Min Precision'),
        mpatches.Patch(color='darkblue', alpha=0.15, label='Max Precision'),
    ]
    ax.legend(handles=handles, loc='best')

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(os.path.dirname(excel_path), f'{book_name}_{fault_level}_precision.png'), dpi=700)

    # Recall variation plot
    fig, ax = plt.subplots(figsize=(15, 12), dpi=100)
    sns.boxplot(y='pair_label', x='recall', data=plot_df, ax=ax, color='salmon', showfliers=False)
    sns.stripplot(y='pair_label', x='recall', data=plot_df, ax=ax, color='darkred', size=6, jitter=True)
    ax.set_title(f'Recall Variation for Isolation Forest Hyperparameters (recall weight = {recall_weight}) : [{fault_level_caption}]', fontsize=11, pad=15)
    ax.set_xlabel('Recall')
    ax.set_ylabel('(n_trees, contam)')

    plt.grid(axis='y')

    for i, label in enumerate(ax.get_yticklabels()):
        if label.get_text().startswith('Group '):
            ax.axhspan(i-0.05, i+0.05, color='black', alpha=0.45)
        if label.get_text() == best_pair:
            ax.axhspan(i-0.3, i+0.3, color='darkgreen', alpha=0.15, zorder=0)

    # Highlight min and max bins
    grouped = plot_df.groupby('pair_label')['recall']
    mean_recalls = grouped.mean()
    min_idx = mean_recalls.idxmin()
    max_idx = mean_recalls.idxmax()
    yticks = ax.get_yticklabels()
    for i, label in enumerate(yticks):
        if label.get_text() == min_idx:
            ax.axhspan(i-0.3, i+0.3, color='darkred', alpha=0.15)
        if label.get_text() == max_idx:
            ax.axhspan(i-0.3, i+0.3, color='darkblue', alpha=0.15)

    handles = [
        mpatches.Patch(color='orange', alpha=0.2, label='h.p. (n_trees, contam)'),
        mpatches.Patch(color='darkgreen', alpha=0.2, label='Best Pair'),
        mpatches.Patch(color='darkred', alpha=0.2, label='Min Recall'),
        mpatches.Patch(color='darkblue', alpha=0.2, label='Max Recall'),
    ]
    ax.legend(handles=handles, loc='best')

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(os.path.dirname(excel_path), f'{book_name}_{fault_level}_recall.png'), dpi=700)
    

if __name__ == "__main__":
    excel_path = os.path.join(os.path.dirname(__file__), 'iswp_1.2', 'analysis', 'iswp_1.2.xlsx')

    IF_histogram_plot(excel_path,
                      
        fault_level = 'subtle', 
        precison_weight = 0.5,
        
        )
    