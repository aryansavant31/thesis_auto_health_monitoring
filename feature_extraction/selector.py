import os, sys

# global imports
from fault_detection.detector import AnomalyDetector
from data.transform import DataNormalizer

# local imports
from .extractor import TimeFeatureExtractor, FrequencyFeatureExtractor, FeatureReducer
from . import tf, ff

# other imports
import numpy as np
import pandas as pd
import inspect
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class FeatureSelector:
    def __init__(self, feat_select_config):
        self.feat_select_config = feat_select_config

        print(f"\nInitializing feature selector for anomaly detection model...")
        if feat_select_config['type'] == 'PCA':
            self.feat_selector = PCA(n_components=feat_select_config['n_comps'])
        elif feat_select_config['type'] == 'LDA':
            self.feat_selector = LDA(n_components=1)

        print(f"\n>> Feature selector initialized with {self.feat_select_config['type'].upper()}")

    def select_features(self, anomaly_detector:AnomalyDetector, data_loader):
        """
        Select features based on importance scores from PCA. 
        """
        print("\nStarting feature selection...")
        data_list = []
        label_list = []

        anomaly_detector.init_input_processors(is_verbose = False)

        for time_data, label, _ in data_loader:
            n_comps = time_data.shape[2]
            n_dims = time_data.shape[3]
            label_np = np.repeat(label.view(-1).numpy(), time_data.size(1))

            data_list.append(time_data)
            label_list.append(label_np)

        time_data_all = torch.vstack(data_list)  # shape (total_samples, n_nodes, n_timesteps, n_dims)
        label_all = np.hstack(label_list)  # shape (total_samples*n_nodes,)
        
        # Domain transform data (mandatory)
        if anomaly_detector._domain == 'time':
            data = anomaly_detector.domain_transformer.transform(time_data_all)

            # Extract features from data
            all_feat_names = [name for name, obj in inspect.getmembers(tf, inspect.isfunction)]
            time_feat_configs = [
                {'type': feat} for feat in all_feat_names
            ]
            feat_data = TimeFeatureExtractor(time_feat_configs).extract(data)

        elif anomaly_detector._domain == 'freq':
            data, freq_bins = anomaly_detector.domain_transformer.transform(time_data_all)

            # Extract features from data
            all_feat_names = [name for name, obj in inspect.getmembers(ff, inspect.isfunction) if name not in ['kp_value_batch', 'get_freq_amp', 'get_freq_psd', 'first_n_modes']]
            freq_feat_configs = [
                {'type': feat} for feat in all_feat_names
            ]
            feat_data = FrequencyFeatureExtractor(freq_feat_configs).extract(data, freq_bins)

             
        # Normalize features
        feat_data_norm = DataNormalizer(norm_type="std").normalize(feat_data)

        # convert np data into pd dataframe
        feat_name_cols = [f"{feat}_{dim}" for feat in all_feat_names for dim in range(n_dims)]

        # if len(feat_name_cols) != n_comps * n_dims:
        #     n_remaining_feats = n_comps * n_dims - len(feat_name_cols)
        #     feat_name_cols.extend([f"ext_feat{idx+1}_dim{dim}" for idx in range(n_remaining_feats) for dim in range(n_dims)])

        feat_data_np = feat_data_norm.view(feat_data_norm.size(0)*feat_data_norm.size(1), feat_data_norm.size(2)*feat_data_norm.size(3)).detach().numpy() # shape (total_samples*n_nodes, n_components*n_dims)
        print("feat_Data_np shape:", feat_data_np.shape)
        feat_data_df = pd.DataFrame(feat_data_np, columns=feat_name_cols)

        # perform feature selection
        if self.feat_select_config['type'] == 'PCA':
            top_feats = self.PCA_based_ranking(feat_data_df)
        elif self.feat_select_config['type'] == 'LDA':
            top_feats = self.LDA_based_ranking(feat_data_df, label_all)

        print(f"\nTop {self.top_n} features selected: {top_feats}")
        print("Feature selection completed. Updating the anomaly detector model to use selected features\n")
        print(75*'-')

        # update anomaly detector with selected features
        anomaly_detector._feat_configs.append({'type': 'from_ranks', 'feat_list': top_feats})

        return anomaly_detector
    

    def PCA_based_ranking(self, feat_data_df):
        # fit PCA model
        self.feat_selector.fit(feat_data_df)

        # Get feature importance scores
        loadings = self.feat_selector.components_  # shape (n_components, n_features)

        abs_loadings = np.abs(loadings)
        loadings_df = pd.DataFrame(abs_loadings.T, index=feat_data_df.columns, columns=[f'PC{i+1}' for i in range(loadings.shape[0])])

        print(f'\n{self.feat_select_config["type"].upper()} loadings are as follows:\n', loadings_df)

        # extract feature name from dimensions
        feat_names = ["_".join(c.split("_")[:-1]) for c in loadings_df.index]

        loadings_df["parent"] = feat_names

        # Aggregate - sum across dimensions and PCs
        feat_importance = (
            loadings_df.groupby("parent")
                    .sum()   # sum across dimensions
                    .sum(axis=1)  # sum across PCs
        )

        self.feat_ranking = feat_importance.sort_values(ascending=False) # feat ranking is panda series (index: feat name, value: importance score)
        self.top_n = self.feat_select_config['n_feats']
        top_feats = self.feat_ranking.head(self.top_n).index.tolist()

        return top_feats

    def LDA_based_ranking(self, feat_data_df, labels):
        # fit LDA model
        self.feat_selector.fit(feat_data_df, labels)

        # Get feature importance scores
        coef = self.feat_selector.coef_[0]  # shape (n_features,)
        feat_names = ["_".join(c.split("_")[:-1]) for c in feat_data_df.columns]

        importance_df = pd.DataFrame({
            'comp': feat_data_df.columns,
            'feature': feat_names,
            'importance': np.abs(coef)
        })

        # Aggregate - sum across dimensions
        self.feat_ranking = (
            importance_df.groupby("feature")['importance']
                    .sum()   # sum across dimensions
                    .sort_values(ascending=False)
        )
        self.top_n = self.feat_select_config['n_feats']
        top_feats = self.feat_ranking.head(self.top_n).index.tolist()

        return top_feats
                             
    ## plots

    def feat_ranking_histogram(self, logger=None):

        if not hasattr(self, 'feat_ranking'):
            raise ValueError("Feature ranking not available. Please run feature_selection() first.")

        model_id = os.path.basename(logger.log_dir) if logger else 'model'
        run_type = 'train'
        tb_tag = model_id.split('-')[0].strip('[]').replace('_(', "  (").replace('+', " + ") if logger else 'feat_selection'

        print("\n" + 12*"<" + " FEATURE RANKING PLOT" + 12*">")
        print(f"\n> Creating feature ranking for {model_id} / {run_type}...")

        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })
        
        plt.figure(figsize=(10, 6))
        feat_rank_plot = sns.barplot(x=self.feat_ranking.values, y=self.feat_ranking.index, palette="viridis")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.title(f"Feature Importance Ranking (Using {self.feat_select_config['type'].upper()})")
        plt.tight_layout()
        plt.show()

        if logger: 
            fig = feat_rank_plot.get_figure()
            fig.savefig(os.path.join(logger.log_dir, f'feat_ranks({model_id}_{run_type}).png'), dpi=500)
            logger.add_figure(f"{tb_tag}/{model_id}/{run_type}/feat_ranks", fig, close=True)
            print(f"\nFeature ranking plot logged at {logger.log_dir}\n")
        else:
            print("\nFeature ranking plot not logged (logger not provided).\n")