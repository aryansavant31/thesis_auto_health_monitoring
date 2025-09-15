import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols

def plot_grid_plot(excel_path):
    # read excel sheet
    df = pd.read_excel(excel_path, sheet_name=0)

    # filter out data for different features
    df = df[df['feats'] == "[]"]

    # Pivot for heatmap
    pivot = df.pivot(index='IF/n_estimators', columns='IF/contam', values='fp')

    plt.figure(figsize=(10, 5))
    plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })
    
    sns.heatmap(pivot, annot=True, cmap='viridis')
    plt.xlabel('contam')
    plt.ylabel('n_estimators')
    plt.title('False Positives by Hyperparameter Combination')
    # save the plot
    plt.savefig(os.path.join(os.path.dirname(__file__), 'iswp_2_heatmap.png'), bbox_inches='tight', dpi=500)

def anova_analysis(excel_path):
    # read excel sheet
    df = pd.read_excel(excel_path, sheet_name=0)

    # filter out data for different features
    df = df[df['feats'] == "[]"]

    df['IF/n_estimators'] = df['IF/n_estimators'].astype(str)
    df['IF/contam'] = df['IF/contam'].astype(str)

    model = ols('fp ~ C(Q("IF/n_estimators")) + C(Q("IF/contam")) + C(Q("IF/n_estimators")):C(Q("IF/contam"))', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    

if __name__ == "__main__":
    excel_path = os.path.join(os.path.dirname(__file__), 'iswp_2_excel.xlsx')
    plot_grid_plot(excel_path)
    # anova_analysis(excel_path)