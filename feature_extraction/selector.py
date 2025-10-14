import os, sys

# global imports
from fault_detection.detector import FaultDetector
from data.transform import DataNormalizer
import copy

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
from collections import Counter

class FeatureSelector:
    def __init__(self, feat_selector_config, n_splits=5):
        self.feat_selector_config = feat_selector_config
        self.n_splits = n_splits

        print(f"\nInitializing feature selector for anomaly detection model...")

        if feat_selector_config['type'] == 'PCA':
            self.feat_selector = PCA(n_components=feat_selector_config['n_comps'])

        elif feat_selector_config['type'] == 'LDA':
            self.feat_selector = LDA(n_components=1)

        print(f"\n>> Feature selector initialized with {self.feat_selector_config['type'].upper()}")

    def split_data(self, data_loader):
        data_list = []
        label_list = []

        for data, label, _ in data_loader:
            n_comps = data.shape[2]
            self.n_dims = data.shape[3]
            label_np = np.repeat(label.view(-1).numpy(), data.size(1))

            data_list.append(data)
            label_list.append(label_np)

        data_all = torch.vstack(data_list)  # shape (total_samples, n_nodes, n_timesteps, n_dims)
        label_all = np.hstack(label_list)  # shape (total_samples*n_nodes,)

        # Split into n_splits
        total_samples = data_all.size(0)
        split_size = total_samples // self.n_splits
        data_splits = [data_all[i*split_size:(i+1)*split_size] if i < self.n_splits-1 else data_all[i*split_size:] for i in range(self.n_splits)]
        label_splits = [label_all[i*split_size:(i+1)*split_size] if i < self.n_splits-1 else label_all[i*split_size:] for i in range(self.n_splits)]

        return data_splits, label_splits
    
    def extract_all_time_features(self, data):
        all_feat_names = [name for name, obj in inspect.getmembers(tf, inspect.isfunction) if name not in ['eo', 'fifth_moment_normalized']]
        time_feat_configs = [
            {'type': feat} for feat in all_feat_names
        ]
        feat_data = TimeFeatureExtractor(time_feat_configs).extract(data)
        return feat_data, all_feat_names
    
    def extract_all_freq_features(self, data, freq_bins):
        all_feat_names = [name for name, obj in inspect.getmembers(ff, inspect.isfunction) if name not in ['kp_value_batch', 'get_freq_amp', 'get_freq_psd', 'first_n_modes', 'kurtosis']]
        freq_feat_configs = [
            {'type': feat} for feat in all_feat_names
        ]
        feat_data = FrequencyFeatureExtractor(freq_feat_configs, self.data_config).extract(data, freq_bins)
        return feat_data, all_feat_names


    def select_features(self, fault_detector:FaultDetector, train_loader, data_config, n_feats=10):
        """
        Select features based on importance scores from PCA. 
        """
        print("\nStarting feature selection...")

        self.data_config = data_config
        self.top_n = n_feats

        # Load all data from data_loader
        time_data_splits, label_splits = self.split_data(train_loader)

        top_feats_all = []
        self.feat_scores_all = {}

        fault_detector.init_input_processors(self.data_config)

        for split_idx in range(self.n_splits):
            time_data = time_data_splits[split_idx]
            label = label_splits[split_idx]
        
        # Domain transform and feature extraction
            if fault_detector.domain == 'time':
                data = fault_detector.domain_transformer.transform(time_data)

                # extract features from data
                feat_data, all_feat_names = self.extract_all_time_features(data)

            elif fault_detector.domain == 'freq':
                data, freq_bins = fault_detector.domain_transformer.transform(time_data)

                # extract features from data
                feat_data, all_feat_names = self.extract_all_freq_features(data, freq_bins)

            elif fault_detector.domain == 'time+freq':
                time_data, freq_mag, freq_bin = fault_detector.domain_transformer.transform(time_data)
                feat_data, all_feat_names = self.extract_all_time_features(time_data)

            # isolate healthy feature data
            feat_data_ok = feat_data[label == 1]
            feat_data_nok = feat_data[label == -1] # debug

        # Normalize features
            if fault_detector.feat_normalizer:
                scalar = copy.deepcopy(fault_detector.feat_normalizer)
                scalar.fit(feat_data_ok)
                feat_data_scaled = scalar.transform(feat_data)
            else:
                raise ValueError("Feature normalizer not found. Please initialize and fit the feature normalizer before feature selection.")
            
            # reshape data to (total_samples*n_nodes, n_components*n_dims)
            feat_data_scaled = feat_data_scaled.view(feat_data_scaled.size(0)*feat_data_scaled.size(1), feat_data_scaled.size(2)*feat_data_scaled.size(3)).detach().numpy()  

            # get feature names
            feat_name_cols = [f"{feat}_{dim}" for feat in all_feat_names for dim in range(self.n_dims)]

            # create dataframe for feature data
            feat_data_df = pd.DataFrame(feat_data_scaled, columns=feat_name_cols)
            # DEBUG
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)

            print(feat_data_df)
            
            #print(feat_data_df.columns[feat_data_df.isna().any()]).tolist()

        # Perform feature ranking
            if self.feat_selector_config['type'] == 'PCA':
                feat_ranking = self.PCA_based_ranking(feat_data_df)
            elif self.feat_selector_config['type'] == 'LDA':
                feat_ranking = self.LDA_based_ranking(feat_data_df, label)


        # Get top n features for this split
            top_feats = list(feat_ranking.index[:n_feats])
            top_feats_all.extend(top_feats)

            for feat in list(feat_ranking.index):
                score = feat_ranking[feat]
                if feat not in self.feat_scores_all:
                    self.feat_scores_all[feat] = {'score': [], 'count': 0}
                self.feat_scores_all[feat]['score'].append(score)
                if feat in top_feats:
                    self.feat_scores_all[feat]['count'] += 1

    # Aggregate top features across splits
        feat_counter = Counter(top_feats_all)
        final_top_feats = [feat for feat, _ in feat_counter.most_common(n_feats)]

        print(f"\nTop {n_feats} consensus features selected: {final_top_feats}")

        # update fault detector with selected features
        fault_detector.feat_configs.append({'type': 'from_ranks', 'feat_list': final_top_feats})

        print("Feature selection completed. Updated the fault detector model to use selected features\n")
        print(75*'-')

        return fault_detector
    

    def PCA_based_ranking(self, feat_data_df):
        # fit PCA model
        self.feat_selector.fit(feat_data_df)

        # Get feature importance scores
        loadings = self.feat_selector.components_  # shape (n_components, n_features)

        abs_loadings = np.abs(loadings)
        loadings_df = pd.DataFrame(abs_loadings.T, index=feat_data_df.columns, columns=[f'PC{i+1}' for i in range(loadings.shape[0])])

        print(f'\n{self.feat_selector_config["type"].upper()} loadings are as follows:\n', loadings_df)

        # extract feature name from dimensions
        feat_names = ["_".join(c.split("_")[:-1]) for c in loadings_df.index]

        loadings_df["parent"] = feat_names

        # Aggregate - sum across dimensions and PCs
        feat_importance = (
            loadings_df.groupby("parent")
                    .mean()   # mean across dimensions
                    .sum(axis=1)  # sum across PCs
        )

        feat_ranking = feat_importance.sort_values(ascending=False) # feat ranking is panda series (index: feat name, value: importance score)
        
        return feat_ranking

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
        feat_ranking = (
            importance_df.groupby("feature")['importance']
                    .mean()   # mean across dimensions
                    .sort_values(ascending=False)
        )
        return feat_ranking
                             
    ## plots

    def feat_ranking_boxplot(self, logger=None):
        if not hasattr(self, 'feat_scores_all'):
            raise ValueError("Feature ranking data not available. Please run select_features() first.")

        model_id = os.path.basename(logger.log_dir) if logger else 'model'
        run_type = 'train'
        tb_tag = model_id.split('-')[0].strip('[]').replace('_(', "  (").replace('+', " + ") if logger else 'feat_selection'

        # Prepare data for plotting
        plot_data = []
        for feat, info in self.feat_scores_all.items():
            for score in info['score']:
                plot_data.append({'feature': feat, 'importance': score, 'count': info['count']})

        plot_df = pd.DataFrame(plot_data)

        # Annotate feature names with count
        feat_counts = {feat: info['count'] for feat, info in self.feat_scores_all.items()}
        annotated_feats = [f"{feat} ({feat_counts[feat]})" for feat in feat_counts]
        sorted_feats = [f"{feat} ({feat_counts[feat]})" for feat in sorted(feat_counts, key=feat_counts.get, reverse=True)]

        # Map original feature names to annotated names
        plot_df['feature_annot'] = plot_df['feature'].map(lambda x: f"{x} ({feat_counts[x]})")
        plot_df['feature_annot'] = pd.Categorical(plot_df['feature_annot'], categories=sorted_feats, ordered=True)

        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })

        plt.figure(figsize=(14, 9))
        ax = plt.gca()
        sns.boxplot(y='feature_annot', x='importance', data=plot_df, ax=ax, color='skyblue', showfliers=False)
        sns.stripplot(y='feature_annot', x='importance', data=plot_df, ax=ax, color='navy', size=6, jitter=True)
        plt.xlabel("Importance Score")
        plt.ylabel(f"Features (count)")
        plt.title(f"Feature Importance Distribution (# feature occurence in top {self.top_n} rank / {self.n_splits} splits) ({self.feat_selector_config['type'].upper()}) : [{model_id}]", pad=15)
        plt.tight_layout()

        if logger:
            plt.savefig(os.path.join(logger.log_dir, f'feat_ranks_variance({model_id}_{run_type}).png'), dpi=500)
            logger.add_figure(f"{tb_tag}/{model_id}/{run_type}/feat_ranks_variance", plt.gcf(), close=True)
            print(f"\nFeature ranking plot with variance logged at {logger.log_dir}\n")
        else:
            plt.show()
            print("\nFeature ranking plot with variance displayed.\n")

    
    def feat_ranking_histogram(self, logger=None):
        if not hasattr(self, 'feat_scores_all'):
            raise ValueError("Feature ranking data not available. Please run select_features() first.")

        model_id = os.path.basename(logger.log_dir) if logger else 'model'
        run_type = 'train'
        tb_tag = model_id.split('-')[0].strip('[]').replace('_(', "  (").replace('+', " + ") if logger else 'feat_selection'

        feat_means = {feat: np.mean(info['score']) for feat, info in self.feat_scores_all.items()}
        feat_counts = {feat: info['count'] for feat, info in self.feat_scores_all.items()}

        # sort features by count
        sorted_feats = sorted(feat_counts, key=feat_counts.get, reverse=True)
        means_sorted = [feat_means[feat] for feat in sorted_feats]
        counts_sorted = [feat_counts[feat] for feat in sorted_feats]
        labels_sorted = [f"{feat} ({feat_counts[feat]})" for feat in sorted_feats]

        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })

        plt.figure(figsize=(14, 9))
        plt.barh(labels_sorted, means_sorted, color='skyblue', edgecolor='navy')
        plt.xlabel("Mean Importance Score")
        plt.ylabel(f"Features (count)")
        plt.title(f"Mean Feature Importance (# feature occurence in top {self.top_n} rank / {self.n_splits} splits) ({self.feat_selector_config['type'].upper()}) : [{model_id}]", pad=15)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if logger:
            plt.savefig(os.path.join(logger.log_dir, f'feat_ranks_hist({model_id}_{run_type}).png'), dpi=500)
            logger.add_figure(f"{tb_tag}/{model_id}/{run_type}/feat_ranks_hist", plt.gcf(), close=True)
            print(f"\nFeature ranking plot with mean score logged at {logger.log_dir}\n")
        else:
            plt.show()
            print("\nFeature ranking plot with mean score displayed.\n")