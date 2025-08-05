import sys
import os

FEATURE_EXTRACTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(FEATURE_EXTRACTION_DIR))


import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.stats import entropy
from feature_extraction.ranking_utils.time_features import *
from feature_extraction.ranking_utils.frequency_features import *
from feature_extraction.ranking_utils.ranking_algorithms import *
import inspect
import ast
import concurrent.futures



# list all the features in a dictionary

def extract_function_names(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    return (function_names)

def load_functions(file_path):
    with open(file_path, 'r') as file:
        exec(file.read(), globals())


file_path_time = os.path.join(FEATURE_EXTRACTION_DIR, "ranking_utils", "time_features.py")
Time_features = extract_function_names(file_path_time)
load_functions(file_path_time)

file_path_frequency = os.path.join(FEATURE_EXTRACTION_DIR, "ranking_utils", "frequency_features.py")
Freq_features = extract_function_names(file_path_frequency)
Freq_features.remove('Kp_value')
load_functions(file_path_frequency)

class FeaturePerformance:
    def __init__(self, Calc_0, Calc_FFT, fftFreq, file_path, n_workers=12):
        """
        Parameters
        ----------
        Calc_0 : dict
            Dictionary containing healthy and unhealthy time signals, with each item being a sample.
        Calc_FFT : dict
            Dictionary containing the healthy and unhealthy frequency amplitudes, with each item being a sample.
        fftFreq : dict
            Dictionary containing the frequencies corresponding to the amplitudes.
        """
        self.Calc_0 = Calc_0
        self.Calc_FFT = Calc_FFT
        self.fftFreq = fftFreq
        self.file_path = file_path
        self.n_workers = n_workers

        if file_path is not None:
            os.makedirs(self.file_path, exist_ok=True)
    
    def rank_time_features(self, bin_number_info, bin_number_chi_square):
        """
        Rank time features based on information gain, Chi-Square, Pearson correlation, and RelieF.
        Parameters
        ----------
        bin_number_info : int
            Number of bins for the information gain
        bin_number_chi_square : int
            Number of bins for the Chi-Square test
        """
        self.info_gain = time_comparison_info_gain(self.Calc_0, bin_number_info, num_workers=self.n_workers)
        self.chi_square = time_comparison_Chi_Square(self.Calc_0, bin_number_chi_square, num_workers=self.n_workers)
        self.pearson = time_comparison_Pearson(self.Calc_0, num_workers=self.n_workers)
        self.relieF = time_comparison_RelieF(self.Calc_0, num_workers=self.n_workers)

        if self.file_path is not None:
            self._save_time_results()

    def rank_frequency_features(self, bin_number_info, bin_number_chi_square):
        """
        Rank frequency features based on information gain, Chi-Square, Pearson correlation, and RelieF.
        Parameters
        ----------
        bin_number_info : int
            Number of bins for the information gain
        bin_number_chi_square : int
            Number of bins for the Chi-Square test
        """
        self.info_gain_freq = freq_comparison_info_gain(self.fftFreq, self.Calc_FFT, bin_number_info, num_workers=self.n_workers)
        self.chi_square_freq = freq_comparison_Chi_Square(self.fftFreq, self.Calc_FFT, bin_number_chi_square, num_workers=self.n_workers)
        self.pearson_freq = freq_comparison_Pearson(self.fftFreq, self.Calc_FFT, num_workers=self.n_workers)
        self.relieF_freq = freq_comparison_RelieF(self.fftFreq, self.Calc_FFT, num_workers=self.n_workers)

        if self.file_path is not None:
            self._save_frequency_results()

    def _save_time_results(self):

        # save pearsion features

        Time_features_Pearson = list(self.pearson.keys())
        Time_values_Pearson = []
        for i in self.pearson.values():
            Time_values_Pearson.append(i[0])  #wE TAKE THE CORRELATION value

        np.save(os.path.join(self.file_path, 'time_values_pearson.npy'), Time_values_Pearson)
        np.save(os.path.join(self.file_path, 'time_features_pearson.npy'), Time_features_Pearson)

        print("\nPearson coefficient for time features is saved")

        plt.figure(figsize=(10, 6))
        bars = plt.barh(Time_features_Pearson, Time_values_Pearson, color='skyblue')
        plt.xlabel('Pearson Coefficient')


        # save chi square features

        Time_features_Chi_Square = list(self.chi_square.keys())
        Time_values_Chi_Square = []
        for i in self.chi_square.values():
            Time_values_Chi_Square.append(i[0])
        np.save(os.path.join(self.file_path, 'time_values_chi.npy'), Time_values_Chi_Square)
        np.save(os.path.join(self.file_path, 'time_features_chi.npy'), Time_features_Chi_Square)

        print("Chi square for time features is saved")

        colors = ['red' if i[1]>0.05 else 'skyblue' for i in self.chi_square.values()]
        plt.figure(figsize=(10, 6))
        bars = plt.barh(Time_features_Chi_Square, Time_values_Chi_Square, color=colors)
        plt.xlabel('Chi-Square Value')


        # save information gain features

        Time_features_info_gain = list(self.info_gain.keys())
        Time_values_info_gain = []
        for i in self.info_gain.values():
            Time_values_info_gain.append(i)
        np.save(os.path.join(self.file_path, 'time_values_info_gain.npy'), Time_values_info_gain)
        np.save(os.path.join(self.file_path, 'time_features_info_gain.npy'), Time_features_info_gain)

        print("Info gain for time features is saved")

        plt.figure(figsize=(10, 6))
        bars = plt.barh(Time_features_info_gain, Time_values_info_gain, color='skyblue')
        plt.xlabel('Information Gain Value')


        # save RelieF features

        Time_features_RelieF = list(self.relieF.keys())
        Time_values_RelieF = []
        for i in self.relieF.values():
            if np.abs(i) < 20:
                Time_values_RelieF.append(i) 

        np.save(os.path.join(self.file_path, 'time_values_relief.npy'), Time_values_RelieF)
        np.save(os.path.join(self.file_path, 'time_features_relief.npy'), Time_features_RelieF)

        print("Relief for time features is saved")

        plt.figure(figsize=(10, 6))
        bars = plt.barh([v for v in self.relieF.keys() if np.abs(self.relieF[v]) < 20], [k for k in self.relieF.values() if np.abs(k)<20], color = 'skyblue')
        #bars = plt.barh(Time_features_RelieF, Time_values_RelieF, color='skyblue')

        plt.xlabel('RelieF weights')

    def _save_frequency_results(self):

        # save pearsion features

        Freq_features_Pearson = list(self.pearson_freq.keys())
        Freq_values_Pearson = []
        for i in self.pearson_freq.values():
            Freq_values_Pearson.append(i[0])
        
        np.save(os.path.join(self.file_path, 'freq_values_pearson.npy'), Freq_values_Pearson)
        np.save(os.path.join(self.file_path, 'freq_features_pearson.npy'), Freq_features_Pearson)

        print("\nPearson coefficient for frequency features is saved")

        plt.figure(figsize=(10, 6))
        colors = ['red' if i[1]>0.05 else 'skyblue' for i in self.pearson_freq.values()]
        bars = plt.barh(Freq_features_Pearson, Freq_values_Pearson, color=colors)
        plt.xlabel('Pearson Coefficient freq') 


        # save chi square features

        Freq_features_Chi = list(self.chi_square_freq.keys())
        Freq_values_Chi = []
        for i in self.chi_square_freq.values():
            Freq_values_Chi.append(i[0])

        np.save(os.path.join(self.file_path, 'freq_values_chi.npy'), Freq_values_Chi)
        np.save(os.path.join(self.file_path, 'freq_features_chi.npy'), Freq_features_Chi)

        print("Chi square for freq features is saved")

        plt.figure(figsize=(10, 6))
        bars = plt.barh(Freq_features_Chi, Freq_values_Chi, color='skyblue')
        plt.xlabel('Chi-Square Value freq')


        # save information gain features

        Freq_features_info_gain = list(self.info_gain_freq.keys())
        Freq_values_info_gain = []
        for i in self.info_gain_freq.values():
            Freq_values_info_gain.append(i)

        np.save(os.path.join(self.file_path, 'freq_values_info_gain.npy'), Freq_values_info_gain)
        np.save(os.path.join(self.file_path, 'freq_features_info_gain.npy'), Freq_features_info_gain)

        print("Info for freq features is saved")

        plt.figure(figsize=(10, 6))
        bars = plt.barh(Freq_features_info_gain, Freq_values_info_gain, color='skyblue')
        plt.xlabel('Information gain')  


        # save RelieF features

        Freq_features_RelieF = list(self.relieF_freq.keys())
        Freq_values_RelieF = []
        for i in self.relieF_freq.values():
            Freq_values_RelieF.append(i)

        np.save(os.path.join(self.file_path, 'freq_values_relief.npy'), Freq_values_RelieF)
        np.save(os.path.join(self.file_path, 'freq_features_relief.npy'), Freq_features_RelieF)

        print("Relief for freq features is saved")

        plt.figure(figsize=(10, 6)) 
        bars = plt.barh([v for v in self.relieF_freq.keys() if np.abs(self.relieF_freq[v]) < 200000], [k for k in self.relieF_freq.values() if np.abs(k)<200000], color = 'skyblue')
        plt.xlabel('RelieF weights freq')

def _info_gain_worker(args):
    Calc_0, feature_name, bin_number = args
    return feature_name, information_gain(Calc_0, globals()[feature_name], bin_number)
        

def time_comparison_info_gain(Calc_0, bin_number, num_workers=12):
    result_time = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_info_gain_worker, [(Calc_0, i, bin_number) for i in Time_features])
        for feature_name, value in results:
            result_time[feature_name] = value
            print(feature_name)

    sorted_result_time = dict(sorted(result_time.items(), key = lambda item: item[1]))
    
    print("\nTime features sorted by information gain")
    return(sorted_result_time)

#We have a second comparison for the frequency features as we need to 
#Take the FFT dictionary into input and there can also be some 2 argument features
#Unlike time features

def _freq_info_gain_worker(args):
    fftFreq, Calc_FFT, feature_name, bin_number = args
    if globals()[feature_name].__code__.co_argcount == 2:
        value = information_gain_freq(Calc_FFT, fftFreq, globals()[feature_name], bin_number)
    else:
        value = information_gain(Calc_FFT, globals()[feature_name], bin_number)
    return feature_name, value

    
def freq_comparison_info_gain(fftFreq, Calc_FFT, bin_number, num_workers=12):
    result_freq = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        args_list = [(fftFreq, Calc_FFT, i, bin_number) for i in Freq_features]
        results = executor.map(_freq_info_gain_worker, args_list)
        for feature_name, value in results:
            result_freq[feature_name] = value
            print(feature_name)
        
    sorted_result_freq = dict(sorted(result_freq.items(), key = lambda item: item[1]))
        
    sorted_result_freq = dict(sorted(result_freq.items(), key = lambda item: item[1]))
    
    print("\nFrequency features sorted by information gain")
    return(sorted_result_freq)


def total_comparison_info_gain(Calc_0, bin_number):
    result_total = {}
    
    for i in Time_features:
        result_total[i] = information_gain(Calc_0, globals()[i], bin_number)
        
    for i in Freq_features:
        result_total[i] = information_gain(Calc_0, globals()[i], bin_number)
    
    sorted_result_total = dict(sorted(result_total.items(), key = lambda item: item[1]))

def _chi_worker(args):
    Calc_0, feature_name, bin_number = args
    return feature_name, Chi_Square(Calc_0, globals()[feature_name], bin_number)
    
def time_comparison_Chi_Square(Calc_0, bin_number, num_workers=12):
    result_time = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_chi_worker, [(Calc_0, i, bin_number) for i in Time_features])
        for feature_name, value in results:
            result_time[feature_name] = value
            print(feature_name)
        
    #We rank the values according to X^2
    sorted_result_time = dict(sorted(result_time.items(), key = lambda item: item[1][0]))
    
    print("\nTime features sorted by Chi-Square")
    return(sorted_result_time)

def _freq_chi_worker(args):
    fftFreq, Calc_FFT, feature_name, bin_number = args
    if globals()[feature_name].__code__.co_argcount == 2:
        value = Chi_Square_freq(Calc_FFT, fftFreq, globals()[feature_name], bin_number)
    else:
        value = Chi_Square(Calc_FFT, globals()[feature_name], bin_number)
    return feature_name, value

def freq_comparison_Chi_Square(fftFreq, Calc_FFT, bin_number, num_workers=12):
    result_freq = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        args_list = [(fftFreq, Calc_FFT, i, bin_number) for i in Freq_features]
        results = executor.map(_freq_chi_worker, args_list)
        for feature_name, value in results:
            result_freq[feature_name] = value
            print(feature_name)

    sorted_result_freq = dict(sorted(result_freq.items(), key = lambda item: item[1]))

    print("\nFrequency features sorted by Chi-Square")
    return(sorted_result_freq)    

def _pearson_worker(args):
    Calc_0, feature_name = args
    return feature_name, Pearson_Corr(Calc_0, globals()[feature_name])

def time_comparison_Pearson(Calc_0, num_workers=12):
    result_time = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_pearson_worker, [(Calc_0, i) for i in Time_features])
        for feature_name, value in results:
            result_time[feature_name] = value
            print(feature_name)

    sorted_result_time = dict(sorted(result_time.items(), key = lambda item: item[1]))
    
    print("\nTime features sorted by Pearson correlation")
    return(sorted_result_time)    

def _freq_pearson_worker(args):
    fftFreq, Calc_FFT, feature_name = args
    if globals()[feature_name].__code__.co_argcount == 2:
        value = Pearson_Corr_freq(Calc_FFT, fftFreq, globals()[feature_name])
        value[0] = np.abs(value[0])
    else:
        value = Pearson_Corr(Calc_FFT, globals()[feature_name])
        value[0] = np.abs(value[0])
    return feature_name, value
    
    
def freq_comparison_Pearson(fftFreq, Calc_FFT, num_workers=12):
    result_freq = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        args_list = [(fftFreq, Calc_FFT, i) for i in Freq_features]
        results = executor.map(_freq_pearson_worker, args_list)
        for feature_name, value in results:
            result_freq[feature_name] = value
            print(feature_name)

    #We sort according to the Pearson correlation coeff, the probabilty needs
    #be checked by hand
    
    sorted_result_freq = dict(sorted(result_freq.items(), key = lambda item: item[1][0]))
    
    print("\nFrequency features sorted by Pearson correlation")
    return(sorted_result_freq)   

def _relief_worker(args):
    Calc_0, feature_name = args
    return feature_name, RelieF(Calc_0, Time_features, globals()[feature_name])

def time_comparison_RelieF(Calc_0, num_workers=12):
    result_time = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_relief_worker, [(Calc_0, i) for i in Time_features])
        for feature_name, value in results:
            result_time[feature_name] = value
            print(feature_name)

    sorted_result_time = dict(sorted(result_time.items(), key = lambda item: item[1]))

    print("\nTime features sorted by RelieF") 
    return(sorted_result_time)  

def _freq_relief_worker(args):
    fftFreq, Calc_FFT, feature_name = args
    if globals()[feature_name].__code__.co_argcount == 2:
        value = RelieF_freq(Calc_FFT, fftFreq, Freq_features, globals()[feature_name])
    else:
        value = RelieF(Calc_FFT, Freq_features, globals()[feature_name])
    return feature_name, value


def freq_comparison_RelieF(fftFreq, Calc_FFT, num_workers=12):
    result_freq = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        args_list = [(fftFreq, Calc_FFT, i) for i in Freq_features]
        results = executor.map(_freq_relief_worker, args_list)
        for feature_name, value in results:
            result_freq[feature_name] = value
            print(feature_name)
   
    #We sort according to the Pearson correlation coeff, the probabilty needs
    #be checked by hand
  
    sorted_result_freq = dict(sorted(result_freq.items(), key = lambda item: item[1]))
    
    print("\nFrequency features sorted by RelieF")
    return(sorted_result_freq)  


# =========================================
# Execution code
# =========================================








