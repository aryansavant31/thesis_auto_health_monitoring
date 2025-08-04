import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None
    
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CONFIG_DIR) if CONFIG_DIR not in sys.path else None

# global imports
from data.config import DataConfig

class FeatureRankingConfig:
    def __init__(self):
        """
        1: Feature Performance Attributes
        ------------------------------
        perf_version : int
            Version of the feature performance.
        bin_num_info : int
            Number of bins for information gain.
        bin_num_chi : int
            Number of bins for chi-square test.

        2: Ranking Attributes
        ------------------------------
        alpha : float
            Gain for feature performance.
        """

    # 1: Feature performance parameters
        self.perf_version   = 1
        self.is_log = True

        self.bin_num_info   = 3
        self.bin_num_chi    = 3

    # 2: Ranking parameters
        self.alpha = 0.5



if __name__ == "__main__":
    from manager import ViewRankings
    ViewRankings().view_ranking_tree()

