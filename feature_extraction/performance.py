import sys
import os

FEATURE_EXTRACTION_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FEATURE_EXTRACTION_DIR))

from feature_extraction.config.manager import FeatureRankingConfigMain
from feature_extraction.ranking_utils.data_adapter import DataAdapter
from data.config import DataConfig
from feature_extraction.ranking_utils.feature_performance import FeaturePerformance

def evaluate_feature_performance():
        rank_config = FeatureRankingConfigMain()
        data_adapter = DataAdapter()
        data_config = DataConfig()

        # load data
        Calc_0, Calc_FFT, fftFreq = data_adapter.get_dictionaries(data_config, run_type='train')

        # get rank log path
        if rank_config.is_log:
            performance_log_path = rank_config.get_performance_log_path(data_config)
        else:
            performance_log_path = None

        # evalaute performance
        feat_perform = FeaturePerformance(
            Calc_0, Calc_FFT, fftFreq, 
            performance_log_path)

        feat_perform.rank_time_features(
            bin_number_info = rank_config.bin_num_info,
            bin_number_chi_square =rank_config.bin_num_chi
        )

        feat_perform.rank_frequency_features(
            bin_number_info = rank_config.bin_num_info,
            bin_number_chi_square = rank_config.bin_num_chi
        )


if __name__ == "__main__":
    evaluate_feature_performance()
    print("\nFeature performance evaluation completed.")
