import sys
import os

RANKING_UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if RANKING_UTILS_DIR not in sys.path:
    sys.path.insert(0, RANKING_UTILS_DIR)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time_features import *
from frequency_features import *
from feature_robustness import test_noise_robustness

import ast

import seaborn as sn
import matplotlib.pyplot as plt


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


def normalisation(amplitudes, interval):
    new = []
    a = interval[0]
    b = interval[1]
    Xmin = np.min(amplitudes)
    Xmax = np.max(amplitudes)
    for i in range(0, len(amplitudes)):
        new.append((b - a) * (amplitudes[i] - Xmin)/(Xmax - Xmin) + a)
    return(new)


def IQR_normalisation(amplitudes, interval):
    
    Q1 = np.percentile(amplitudes, 25)
    Q3 = np.percentile(amplitudes, 75)
    IQR = Q3 - Q1
    
    
    #Sometimes when we have a small variety of values, (1 2 or 3 different ones only)
    #Q1 and Q3 will be the same and the usual algorithm will return a constant
    #list that cannot be normalised then
    #To prevent this we directly normalise it (low chance of absurd values in those cases)
    if Q1 == Q3:
        return(normalisation(amplitudes, interval))
    
    #In statistics we use 1.5 times the IQR to determine outliers
    low_bound = Q1 - 1.5 * IQR
    high_bound = Q3 + 1.5 * IQR
    
    filtered_ampli = np.clip(amplitudes, low_bound, high_bound)

    return(normalisation(filtered_ampli, interval))


class FinalFeatureRanking:
    def __init__(self, Calc_10, ranking_path):
        self.ranking_path = ranking_path
        self.file_path_util = os.path.dirname(ranking_path)

        self.Calc_10 = Calc_10

    def _load_util_files(self):
        self.Freq_values_Chi = np.load(os.path.join(self.file_path_util, 'Freq_values_Chi.npy'))
        self.Freq_features_Chi = np.load(os.path.join(self.file_path_util, 'Freq_features_Chi.npy'))

        self.Freq_values_RelieF = np.load(os.path.join(self.file_path_util, 'Freq_values_RelieF.npy'))
        self.Freq_features_RelieF = np.load(os.path.join(self.file_path_util, 'Freq_features_RelieF.npy'))

        self.Freq_values_info = np.load(os.path.join(self.file_path_util, 'Freq_values_info.npy'))
        self.Freq_features_info = np.load(os.path.join(self.file_path_util, 'Freq_features_info.npy'))

        self.Freq_values_Pearson = np.load(os.path.join(self.file_path_util, 'Freq_values_Pearson.npy'))
        self.Freq_features_Pearson = np.load(os.path.join(self.file_path_util, 'Freq_features_Pearson.npy'))


        self.Time_values_Pearson = np.load(os.path.join(self.file_path_util,'Time_values_Pearson.npy'))
        self.Time_features_Pearson = np.load(os.path.join(self.file_path_util,'Time_features_Pearson.npy'))

        self.Time_values_Chi = np.load(os.path.join(self.file_path_util,'Time_values_Chi.npy'))
        self.Time_features_Chi = np.load(os.path.join(self.file_path_util,'Time_features_Chi.npy'))

        self.Time_values_Info_Gain = np.load(os.path.join(self.file_path_util,'Time_values_Info_Gain.npy'))
        self.Time_features_Info_Gain = np.load(os.path.join(self.file_path_util,'Time_features_Info_Gain.npy'))

        self.Time_values_RelieF = np.load(os.path.join(self.file_path_util, 'Time_values_RelieF.npy'))
        self.Time_features_RelieF = np.load(os.path.join(self.file_path_util, 'Time_features_RelieF.npy'))

    def get_correlation_matrix(self):
        result_feature = {}
        for i in Time_features:
            result_feature[i] = []
            for key in self.Calc_10.keys():
                feature = globals()[i](self.Calc_10[key])
                if isinstance(feature, (list, np.ndarray, tuple)):
                    result_feature[i].append(feature[0])
                else:
                    result_feature[i].append(feature)

        df = pd.DataFrame(result_feature, columns = result_feature.keys())

        corrMatrix = df.corr()
        n = np.shape(corrMatrix)[0]
        count_90 = 0
        count_75 = 0
        for i in range(0, n):
            for j in range(0, n):
                corrMatrix.iloc[i, j] = np.abs(corrMatrix.iloc[i, j])
                if np.abs(corrMatrix.iloc[i, j])>= 0.9:
                    count_90 += 1
                if np.abs(corrMatrix.iloc[i, j])>= 0.75:
                    count_75 += 1

        sn.heatmap(corrMatrix, vmin= 0.9, vmax = 1)
        plt.plot()

    def get_final_ranking(self):

        # load utility files
        self._load_util_files()

        #Healthy_Chi_10
        Time_values_Chi = normalisation(Time_values_Chi, [0,1])
        Freq_values_Chi = IQR_normalisation(Freq_values_Chi, [0,1])

        #Healthy_Info_10
        Time_values_Info_Gain = normalisation(Time_values_Info_Gain, [0,1])
        Freq_values_info = IQR_normalisation(Freq_values_info, [0,1])
        #Healthy_Pearson_10
        Time_values_Pearson = normalisation(Time_values_Pearson, [0,1])
        Freq_values_Pearson = IQR_normalisation(Freq_values_Pearson, [0,1])

        #Healthy_RelieF_10
        Time_values_RelieF = normalisation(Time_values_RelieF, [0,1])
        Freq_values_RelieF = IQR_normalisation(Freq_values_RelieF, [0,1]) 

        #To do our final ranking we calculate to weighted means : first the one 
        #comparing the 4 ranking algorithms, and secondly the one comparing
        #the robustness criterias

        Final_Ranking_feature = {}
        Final_Ranking_Robustness = {}
        Final_Ranking_Time = {}
        Final_Ranking_Freq = {}

        for feature in Time_features:
            Final_Ranking_feature[feature] = 0
            Final_Ranking_Robustness[feature] = 0
        
        for feature in Freq_features:
            Final_Ranking_feature[feature] = 0
            Final_Ranking_Robustness[feature] = 0
        
        #We convert the keys into lists to be able to access them below
        #Even though the total list of features is the same through the 4 algorithms, 
        #the order is different because of the feature ranking proper to each alogrithm
        #So to keep each score to its own feature we used a separate list every time
        # We didn't keep dictionnaries as list are easier to manipulate for operations
        #(You can't access dictionary keys  in d.keys() like a list for instance because
        #of the dictionary having its own class/structure in Python code)  


        Time_features_Chi = list(Time_features_Chi)    
        Time_features_Pearson = list(Time_features_Pearson)   
        Time_features_Info_Gain = list(Time_features_Info_Gain)   
        Time_features_RelieF = list(Time_features_RelieF)   
            
        #Same thing for the frequency features
        
        Freq_features_Pearson = list(Freq_features_Pearson)
        Freq_features_info = list(Freq_features_info)
        Freq_features_RelieF = list(Freq_features_RelieF)
        Freq_features_Chi = list(Freq_features_Chi)


        for i in range(0, len(Time_values_Chi)):
            Final_Ranking_feature[Time_features_Chi[i]] = Final_Ranking_feature[Time_features_Chi[i]] + Time_values_Chi[i]/4
            
        for i in range(0, len(Time_values_Pearson)):
            Final_Ranking_feature[Time_features_Pearson[i]] = Final_Ranking_feature[Time_features_Pearson[i]] + Time_values_Pearson[i]/4
            
        for i in range(0, len(Time_values_Info_Gain)):
            Final_Ranking_feature[Time_features_Info_Gain[i]] = Final_Ranking_feature[Time_features_Info_Gain[i]] + Time_values_Info_Gain[i]/4
            
        for i in range(0, len(Time_values_RelieF)):
            Final_Ranking_feature[Time_features_RelieF[i]] = Final_Ranking_feature[Time_features_RelieF[i]] + Time_values_RelieF[i]/4
            
        for i in range(0, len(Freq_values_Chi)):
            Final_Ranking_feature[Freq_features_Chi[i]] = Final_Ranking_feature[Freq_features_Chi[i]] + Freq_values_Chi[i]/4
            #print(Freq_values_Chi[i]/4)
            
        for i in range(0, len(Freq_values_RelieF)):
            Final_Ranking_feature[Freq_features_RelieF[i]] = Final_Ranking_feature[Freq_features_RelieF[i]] + Freq_values_RelieF[i]/4
            
        for i in range(0, len(Freq_values_info)):
            Final_Ranking_feature[Freq_features_info[i]] = Final_Ranking_feature[Freq_features_info[i]] + Freq_values_info[i]/4
            
        for i in range(0, len(Freq_values_Pearson)):
            # print(i)
            # print(Freq_values_Pearson[i])
            Final_Ranking_feature[Freq_features_Pearson[i]] = Final_Ranking_feature[Freq_features_Pearson[i]] + Freq_values_Pearson[i]/4
    

        #Then it becomes a question of tresholds for the noise/drift/offset resilience etc ...
        #Let's say we put the resilience threshold to be at 1% at 40 dB
        #Then we create a binary variable of 1 if the treshold is respected 0 else
        #We'll here treat every criteria as being the same importance, noise/offset, etc ...
        #In truth, the noise could be augmented in its importance (maybe 3/1/1/1 ?)

        Resilient_30dB_1pct = test_noise_robustness()

        for feature in Freq_features + Time_features:
            Final_Ranking_Robustness[feature] = Final_Ranking_Robustness[feature] + Resilient_30dB_1pct[feature]
            #print([Final_Ranking_Robustness[feature], Resilient_30dB_1pct[feature]])



        #Now we just need to choose the weight between performance and robustness
        #Here we chose a weight of 0.7, favoring performance slightly better

        alpha = 0.7

        for feature in Time_features:
            Final_Ranking_Time[feature] = alpha * Final_Ranking_feature[feature] + (1 - alpha) * Final_Ranking_Robustness[feature]

        for feature in Freq_features:
            Final_Ranking_Freq[feature] = alpha * Final_Ranking_feature[feature] + (1 - alpha) * Final_Ranking_Robustness[feature]

        Final_Ranking_Time = dict(sorted(Final_Ranking_Time.items(), key = lambda item: item[1]))
        Final_Ranking_Freq = dict(sorted(Final_Ranking_Freq.items(), key = lambda item: item[1]))
        Final_Ranking_feature = dict(sorted(Final_Ranking_feature.items(), key = lambda item: item[1]))    
            
        version = os.path.basename(self.file_path_util).split('_')[-1]
        np.save(os.path.join(self.ranking_path, f'time_feature_ranking_v{version}.npy'), Final_Ranking_Time)
        np.save(os.path.join(self.ranking_path, f'freq_feature_ranking_v{version}.npy'), Final_Ranking_Freq)