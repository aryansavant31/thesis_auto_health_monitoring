import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics_from_excel(excel_path):
    """
    Plots grouped bar charts of test accuracy, precision, and recall vs. noise for each unique feature in the Excel sheet.
    """
    # Read the first sheet
    df = pd.read_excel(excel_path, sheet_name=0)
    
    # Clean feature names for grouping (remove brackets and spaces)
    def clean_feat(feat):
        if isinstance(feat, str):
            return feat.strip('[]').replace("'", "").replace('"', '').replace(' ', '')
        return str(feat)
    
    df['feats_clean'] = df['feats'].apply(clean_feat)
    
    # Get unique features
    unique_feats = df['feats_clean'].unique()
    
    for idx, feat in enumerate(unique_feats):
        if feat == '':
            feat_rows = df[(df['feats_clean'] == feat) & (df['domain'] == 'time(cutoff_freq=0)')]
        else:
            feat_rows = df[df['feats_clean'] == feat]
        
        # Extract required columns
        noise = feat_rows['noise']
        test_acc = feat_rows['test_accuracy']
        precision = feat_rows['precision']
        recall = feat_rows['recall']
        
        # Bar width and positions
        x = range(len(noise))
        bar_width = 0.2
        
        plt.figure(figsize=(8, 5))

        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })

        plt.bar([i - bar_width for i in x], test_acc, width=bar_width, label='Test Accuracy', alpha=0.8)
        plt.bar(x, precision, width=bar_width, label='Precision', alpha=0.8)
        plt.bar([i + bar_width for i in x], recall, width=bar_width, label='Recall', color='grey', alpha=0.8)
        
        plt.xlabel('Sigma of Noise')
        plt.ylabel('Metric Value')
        # Title: remove brackets, keep elements, capitalize
        title = feat.replace('[', '').replace(']', '').replace("'", "").replace('"', '').replace(' ', '')
        title = title.replace(',', ', ')
        if title == '':
            title = 'No Features'

        plt.title(title.title())
        plt.xticks(x, noise)
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"noise_({feat}).png"
        plt.savefig(os.path.join(os.path.dirname(excel_path), plot_filename), dpi=500)
        plt.close()

if __name__ == "__main__":
    excel_path = os.path.join(os.path.dirname(__file__), 'iswp1_excel.xlsx')
    plot_metrics_from_excel(excel_path)