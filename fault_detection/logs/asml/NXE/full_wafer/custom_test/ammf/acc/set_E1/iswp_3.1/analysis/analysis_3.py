import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.patches as mpatches


def feature_histogram_plot(excel_path, fault_level, precison_weight=0.5):
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

    # Use the 'feat' column directly as label
    filtered_df = filtered_df.sort_values(['feats'])
    plot_df = filtered_df[['feats', 'precision', 'recall']].copy()
    plot_df.rename(columns={'feats': 'feature'}, inplace=True)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    # Find best overall metric feature (highest mean of precision and recall)
    overall_means = plot_df.groupby('feature').apply(
        lambda g: precison_weight * g['precision'].mean() + recall_weight * g['recall'].mean()
    )
    best_feature = overall_means.idxmax()

    # Precision variation plot
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    sns.boxplot(y='feature', x='precision', data=plot_df, ax=ax, color='skyblue', showfliers=False)
    sns.stripplot(y='feature', x='precision', data=plot_df, ax=ax, color='navy', size=6, jitter=True)
    ax.set_title(f'Precision Variation for Features (precision weight = {precison_weight}) : [{fault_level_caption}]', fontsize=11, pad=15)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Features')
    ax.set_xlim(-0.05, 1.05)

    plt.grid(axis='y')

    for i, label in enumerate(ax.get_yticklabels()):
        if label.get_text() == best_feature:
            ax.axhspan(i-0.3, i+0.3, color='darkgreen', alpha=0.15, zorder=0)

    # # Highlight min and max bins
    # grouped = plot_df.groupby('feature')['precision']
    # mean_precisions = grouped.mean()
    # min_idx = mean_precisions.idxmin()
    # max_idx = mean_precisions.idxmax()
    # yticks = ax.get_yticklabels()
    # for i, label in enumerate(yticks):
    #     if label.get_text() == min_idx:
    #         ax.axhspan(i-0.3, i+0.3, color='darkred', alpha=0.15)
    #     if label.get_text() == max_idx:
    #         ax.axhspan(i-0.3, i+0.3, color='darkblue', alpha=0.15)

    handles = [
        mpatches.Patch(color='darkgreen', alpha=0.15, label='Best Feature'),
        # mpatches.Patch(color='darkred', alpha=0.15, label='Min Precision'),
        # mpatches.Patch(color='darkblue', alpha=0.15, label='Max Precision'),
    ]
    ax.legend(handles=handles, loc='best')

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(os.path.dirname(excel_path), f'{book_name}_{fault_level}_fault_feature_precision.png'), dpi=700)

    # Recall variation plot
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    sns.boxplot(y='feature', x='recall', data=plot_df, ax=ax, color='salmon', showfliers=False)
    sns.stripplot(y='feature', x='recall', data=plot_df, ax=ax, color='darkred', size=6, jitter=True)
    ax.set_title(f'Recall Variation for Features (recall weight = {recall_weight}) : [{fault_level_caption}]', fontsize=11, pad=15)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Features')
    ax.set_xlim(-0.05, 1.05)

    plt.grid(axis='y')

    for i, label in enumerate(ax.get_yticklabels()):
        if label.get_text() == best_feature:
            ax.axhspan(i-0.3, i+0.3, color='darkgreen', alpha=0.15, zorder=0)

    # # Highlight min and max bins
    # grouped = plot_df.groupby('feature')['recall']
    # mean_recalls = grouped.mean()
    # min_idx = mean_recalls.idxmin()
    # max_idx = mean_recalls.idxmax()
    # yticks = ax.get_yticklabels()
    # for i, label in enumerate(yticks):
    #     if label.get_text() == min_idx:
    #         ax.axhspan(i-0.3, i+0.3, color='darkred', alpha=0.15)
    #     if label.get_text() == max_idx:
    #         ax.axhspan(i-0.3, i+0.3, color='darkblue', alpha=0.15)

    handles = [
        mpatches.Patch(color='darkgreen', alpha=0.2, label='Best Feature'),
        # mpatches.Patch(color='darkred', alpha=0.2, label='Min Recall'),
        # mpatches.Patch(color='darkblue', alpha=0.2, label='Max Recall'),
    ]
    ax.legend(handles=handles, loc='best')

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(os.path.dirname(excel_path), f'{book_name}_{fault_level}_fault_feature_recall.png'), dpi=700)
    

if __name__ == "__main__":
    excel_path = os.path.join(os.path.dirname(__file__), 'iswp_3.1.xlsx')

    feature_histogram_plot(excel_path,
                      
        fault_level = 'subtle', 
        precison_weight = 0.5,
        
        )
    