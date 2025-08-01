import sys
import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

FEX_DIR = os.path.join((os.path.abspath(__file__)))
sys.path.insert(0, FEX_DIR) if FEX_DIR not in sys.path else None

# global imports
from data.settings import DataConfig

# local imports
from config.manager import FeatureRankingManager
from ranking_utils.data_adapter import DataAdapter
from ranking_utils.feature_performance import FeaturePerformance

class PerformanceMain:
    def __init__(self):
        self.data_adapter = DataAdapter()
         
    def evaluate_feature_performance(self, data_config:DataConfig, rank_config:FeatureRankingManager):
   
            # load data
            Calc_0, Calc_FFT, fftFreq = self.data_adapter.get_dictionaries(data_config)

            # get rank log path
            if rank_config.is_log:
                perf_log_path = rank_config.get_perf_log_path(data_config)
                rank_config.save_perf_id()
            else:
                perf_log_path = None

            # evalaute performance
            feat_perform = FeaturePerformance(
                Calc_0, Calc_FFT, fftFreq, 
                perf_log_path)

            feat_perform.rank_time_features(
                bin_number_info = rank_config.bin_num_info,
                bin_number_chi_square =rank_config.bin_num_chi
            )

            feat_perform.rank_frequency_features(
                bin_number_info = rank_config.bin_num_info,
                bin_number_chi_square = rank_config.bin_num_chi
            )


if __name__ == "__main__":
    data_config = DataConfig(run_type='custom_test')
    rank_config = FeatureRankingManager()
    perf_main = PerformanceMain()

    perf_main.evaluate_feature_performance(data_config, rank_config)

    print("\nFeature performance evaluation completed.")
