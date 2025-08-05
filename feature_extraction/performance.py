import sys
import os
import warnings
# force warnings to always be displayed and go to stderr
warnings.filterwarnings('always')

ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

FEX_DIR = os.path.join((os.path.abspath(__file__)))
sys.path.insert(0, FEX_DIR) if FEX_DIR not in sys.path else None

# global imports
from data.config import DataConfig

# local imports
from settings.manager import FeatureRankingManager
from ranking_utils.data_adapter import DataAdapter
from ranking_utils.feature_performance import FeaturePerformance
from ranking_utils.console_logger import ConsoleLogger

class PerformanceMain:
    def __init__(self, data_config:DataConfig):
        self.data_adapter = DataAdapter(data_config)
         
    def evaluate_feature_performance(self, rank_config:FeatureRankingManager):
   
            # load data
            Calc_0, Calc_FFT, fftFreq = self.data_adapter.get_dictionaries()

            # get rank log path
            if rank_config.is_log:
                perf_log_path = rank_config.get_perf_log_path()
            else:
                perf_log_path = None

            # evalaute performance
            feat_perform = FeaturePerformance(
                Calc_0, Calc_FFT, fftFreq, 
                perf_log_path, n_workers=rank_config.n_workers)

            feat_perform.rank_time_features(
                bin_number_info = rank_config.bin_num_info,
                bin_number_chi_square =rank_config.bin_num_chi
            )

            feat_perform.rank_frequency_features(
                bin_number_info = rank_config.bin_num_info,
                bin_number_chi_square = rank_config.bin_num_chi
            )
            return perf_log_path


if __name__ == "__main__":
    # create console logger to log all the outputs in terminal
    console_logger = ConsoleLogger()

    # capture and store console output in memory
    with console_logger.capture_output():
        print("\nStarting feature performance evaluation...")

        data_config = DataConfig(run_type='custom_test')
        rank_config = FeatureRankingManager(data_config)
        perf_main = PerformanceMain(data_config)

        log_path = perf_main.evaluate_feature_performance(rank_config)

        print("\nFeature performance evaluation completed.")

    # if loggin enabled, save the captured output to a file
    if rank_config.is_log:
        file_path = os.path.join(log_path, "console_output.txt")
        base_name = os.path.basename(log_path)
        console_logger.save_to_file(file_path, script_name="performance.py", base_name=base_name)
