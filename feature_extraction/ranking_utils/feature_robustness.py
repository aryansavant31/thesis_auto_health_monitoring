import sys
import os

RANKING_UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if RANKING_UTILS_DIR not in sys.path:
    sys.path.insert(0, RANKING_UTILS_DIR)

import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.stats import entropy
from time_features import *
from frequency_features import *
from scipy.stats import kendalltau
import ast
from scipy.stats import pearsonr

import seaborn as sn
import matplotlib.pyplot as plt
import concurrent.futures


def extract_function_names(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    return (function_names)

def load_functions(file_path):
    with open(file_path, 'r') as file:
        exec(file.read(), globals())


file_path_time = os.path.join(RANKING_UTILS_DIR, "time_features.py")
Time_features = extract_function_names(file_path_time)
load_functions(file_path_time)

file_path_frequency = os.path.join(RANKING_UTILS_DIR, "frequency_features.py")
Freq_features = extract_function_names(file_path_frequency)
Freq_features.remove('Kp_value')
load_functions(file_path_frequency)


def coefficient_variation(signal):
    return(np.std(signal)/np.mean(signal) if np.mean(signal) != 0 else np.nan)


def SNR(signal, noise):
    
    A = np.sum(np.abs(signal) ** 2)
    B = np.sum(np.abs(noise) ** 2)
    return(10*np.log10(A/B))


def sigma_finder_SNR(signal, SNR):
    RMS_signal = np.sqrt(np.sum(np.abs(signal) ** 2)/(len(signal)))
    K = 10 ** (SNR/20)
    
    return(RMS_signal/K)

def compute_noise_cv_for_feature(args):
    feature, Freq_features, Time_features, AAA, AAAfreq, range_SNR = args
    result_feature_noise_SNR = {}
    noise_CV_SNR = []
    for i in range(0, len(range_SNR)):
        sigma_SNR = sigma_finder_SNR(AAA, range_SNR[i])
        noised_signals_real = []
        noised_signals_freq = []
        for p in range(0, 20):
            noised_signals_real.append(AAA + np.random.normal(0, sigma_SNR, AAA.shape[0]))
            noised_signals_freq.append(AAAfreq + np.random.normal(0, sigma_SNR, AAAfreq.shape[0]))
        result_feature_noise_SNR[feature] = []
        for p in range(0, 20):
            if globals()[feature].__code__.co_argcount == 2:
                val = globals()[feature](AAAfreq, noised_signals_freq[p])
                if isinstance(val, (list, tuple, np.ndarray)):
                    result_feature_noise_SNR[feature].append(val[0])
                elif isinstance(val, (int, float, complex)):
                    result_feature_noise_SNR[feature].append(val)
            else:
                val = globals()[feature](noised_signals_real[p])
                if isinstance(val, (list, tuple, np.ndarray)):
                    result_feature_noise_SNR[feature].append(val[0])
                elif isinstance(val, (int, float, complex)):
                    result_feature_noise_SNR[feature].append(val)
        cv = coefficient_variation(result_feature_noise_SNR[feature])
        if np.isnan(cv):
            noise_CV_SNR.append(0)
        else:
            noise_CV_SNR.append(100 * np.abs(cv))
    return feature, noise_CV_SNR


def test_noise_robustness():
    #Test simple sine

    A0 = np.linspace(0, 100, 1000)
    AAA = np.sin(A0)


    #FFT calculation for this sinus to be able to calculate the ranking with frequency features too
    spacing = 1/48000

    AAAfft = fft(AAA, norm='forward')

    AAAfreq = np.fft.rfftfreq(len(AAAfft), spacing)
    AAAfft = AAAfft[:len(AAAfft)//2 +1]

    """
    #Example to see how the behaviour changes with the feature when the signal becomes complex
    #AAA = np.sin(2*A0) + np.sin(5*A0) + np.sin(7*A0)
    """

    noise = np.random.normal(0, sigma_finder_SNR(AAA, 20), AAA.shape[0])
    SNR_sine = SNR(AAA, noise)


    #We've taken a range from 10 db (worst performance allowed) to 40 db (highest performance 
    #considered, above that the SNR 'll be so high/good that noise resiliency doesn't matter).
    range_SNR = np.linspace(10,40,155)

    #Then we find the sigma associated to this for our signal and create the noise
    #associated to this sigma


    #In our example the accelearions are normalized between 0 and 1 so I'll
    #Consider an offset 

    #In the previous method we took small ranges around the values of SNR (+-2%) 
    #to create a range to see a variation and obtain a relevant CV
    #A more pertinent approach is to fix ourselves at the wanted SNR level and take
    #20 different realisations of noise for this noise level and see the variation
    #of our signal for those different noises (but same SNR)

    #drifted_signals = {}
    features = Freq_features + Time_features
    args_list = [(feature, Freq_features, Time_features, AAA, AAAfreq, range_SNR) for feature in features]
    noise_CV_SNR = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(compute_noise_cv_for_feature, args_list))

    for feature, cv_list in results:
        noise_CV_SNR[feature] = cv_list
        # plt.figure()
        # plt.plot(range_SNR, noise_CV_SNR[feature])
        # plt.title('SNR/CV evolution ' + feature)
        # plt.xlabel('SNR value (dB)')
        # plt.ylabel('Coefficient of variation (%)')

    #Test to know the top 5 best features at 30 dB ( the most resilient to the noise)
    #With the current range_SNR, 30 dB is attained at the 102th index

    Thirty_db_index = 103
    Fourty_db_index = 154


    #The ranking dictionnaries contain all the values of the feature that suffice the 
    #chosen criteria to be able to see the best performing ones

    #The "Resilient" dictionaries contain 0 or 1 if the feature is resilient or not
    #And are useful in the last global ranking done below
    Ranking_30dB_5pct = {}
    Ranking_30dB_1pct = {}
    Ranking_40dB_1pct = {}
    Resilient_40dB_1pct = {}
    Resilient_30dB_1pct = {}

    for feature in Freq_features + Time_features:
        Resilient_40dB_1pct[feature] = 0
        Resilient_30dB_1pct[feature] = 0
        
        
    for feature in Freq_features + Time_features:
        if noise_CV_SNR[feature][Thirty_db_index] <=1:
            Ranking_30dB_1pct[feature] = noise_CV_SNR[feature][Thirty_db_index]
            Resilient_30dB_1pct[feature] = 1
        else:
            Resilient_30dB_1pct[feature] = 0
            
        if noise_CV_SNR[feature][Thirty_db_index] <=5:
            Ranking_30dB_5pct[feature] = noise_CV_SNR[feature][Thirty_db_index]
            
        if noise_CV_SNR[feature][Fourty_db_index] <=1:
            Ranking_40dB_1pct[feature] = noise_CV_SNR[feature][Fourty_db_index]
            Resilient_40dB_1pct[feature] = 1
        else:
            Resilient_40dB_1pct[feature] = 0
            
    Sorted_Ranking_30dB_5pct = dict(sorted(Ranking_30dB_5pct.items(), key=lambda item: item[1]))

    Sorted_Ranking_30dB_1pct = dict(sorted(Ranking_30dB_1pct.items(), key=lambda item: item[1]))

    Sorted_Ranking_40dB_1pct = dict(sorted(Ranking_40dB_1pct.items(), key=lambda item: item[1]))

    return Resilient_30dB_1pct

