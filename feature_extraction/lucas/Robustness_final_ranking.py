# -*- coding: utf-8 -*-
"""
Created on Wed May  7 12:08:59 2025

@author: lfernan4
"""

import scipy.io
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.stats import entropy
from Time_features import *
from Frequency_features import *
from scipy.stats import kendalltau
print(1)
print(2)
import ast
from scipy.stats import pearsonr


#Calc_10 = segmenting_healthy(Calc_0, 10)
# Healthy_Chi_10 = time_comparison_Chi_Square(Calc_10[0], 3)
# Healthy_Pearson_10 = time_comparison_Pearson(Calc_10[0])
# Healthy_Info_10 = time_comparison_info_gain(Calc_10[0], 3)
#%%


Calc_10 = np.load('Calc_10.npy', allow_pickle=True)

Freq_values_Chi = np.load('Freq_values_Chi.npy')
Freq_features_Chi = np.load('Freq_features_Chi.npy')

Freq_values_RelieF = np.load('Freq_values_RelieF.npy')
Freq_features_RelieF = np.load('Freq_features_RelieF.npy')

Freq_values_info = np.load('Freq_values_info.npy')
Freq_features_info = np.load('Freq_features_info.npy')

Freq_values_Pearson = np.load('Freq_values_Pearson.npy')
Freq_features_Pearson = np.load('Freq_features_Pearson.npy')




Time_values_Pearson = np.load('Time_values_Pearson.npy')
Time_features_Pearson = np.load('Time_features_Pearson.npy')

Time_values_Chi = np.load('Time_values_Chi.npy')
Time_features_Chi = np.load('Time_features_Chi.npy')

Time_values_Info_Gain = np.load('Time_values_Info_Gain.npy')
Time_features_Info_Gain = np.load('Time_features_Info_Gain.npy')

Time_values_RelieF = np.load('Time_values_RelieF.npy')
Time_features_RelieF = np.load('Time_features_RelieF.npy')




def extract_function_names(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    return (function_names)




def load_functions(file_path):
    with open(file_path, 'r') as file:
        exec(file.read(), globals())


# Exemple d'utilisation
file_path_time = "Time_features.py"
Time_features = extract_function_names(file_path_time)
load_functions(file_path_time)

file_path_frequency = "Frequency_features.py"
Freq_features = extract_function_names(file_path_frequency)
Freq_features.remove('Kp_value')
load_functions(file_path_frequency)


#%%
#This is the part where we calculate the matrix with the cross correlation 
#between features

print(3)
result_feature = {}
for i in Time_features:
    result_feature[i] = []
    for key in Calc_10[0].keys():
        result_feature[i].append(globals()[i](Calc_10[0][key])[0])

    

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


#print(corrMatrix)

import seaborn as sn
import matplotlib.pyplot as plt

sn.heatmap(corrMatrix, vmin= 0.9, vmax = 1)
plt.plot()
print(3.5)

#%%


#kendall = kendalltau(Time_values_Info_Gain, Time_values_Pearson)



def coefficient_variation(signal):
    return(np.std(signal)/np.mean(signal) if np.mean(signal) != 0 else np.nan)


def normalisation(amplitudes, interval):
    new = []
    a = interval[0]
    b = interval[1]
    Xmin = np.min(amplitudes)
    Xmax = np.max(amplitudes)
    for i in range(0, len(amplitudes)):
        new.append((b - a) * (amplitudes[i] - Xmin)/(Xmax - Xmin) + a)
    return(new)




#%%


#Tests SNR curves


def SNR(signal, noise):
    
    A = np.sum(np.abs(signal) ** 2)
    B = np.sum(np.abs(noise) ** 2)
    return(10*np.log10(A/B))



#This function finds the sigma of the white noise to apply to the signal 
#To obtain the wanted SNR (given in input in decibels)

def sigma_finder_SNR(signal, SNR):
    RMS_signal = np.sqrt(np.sum(np.abs(signal) ** 2)/(len(signal)))
    K = 10 ** (SNR/20)
    
    return(RMS_signal/K)
    

#Test simple sine
    
A0 = np.linspace(0, 100, 1000)
AAA = np.sin(A0)


#FFT calculation for this sinus to be able to calculate the ranking with frequency features too
spacing = 1/48000

AAAfft = fft(AAA)

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
sigma_SNR = []
CV_values = {}
result_feature_noise_SNR = {}
noise_CV_SNR = {}
#drift_CV_SNR = {}

print(4)

    
print(5)
for feature in Freq_features + Time_features:
    print(feature)
    noise_CV_SNR[feature] = []
    for i in range(0, len(range_SNR)):
        sigma_SNR = sigma_finder_SNR(AAA, range_SNR[i])
        noised_signals_real = []
        noised_signals_freq = []
        #drifted_signals[i] = []
        
        for p in range(0, 20):
            noised_signals_real.append(AAA + np.random.normal(0, sigma_SNR, AAA.shape[0]))
            noised_signals_freq.append(AAAfreq + np.random.normal(0, sigma_SNR, AAAfreq.shape[0]))
        
       
        
        #print(i)
        result_feature_noise_SNR[feature] = []
        #print(globals()[i](range_value))
        #print(j)
        for p in range(0, 20):
            
            #We take into account the frequency features with 2 qrguments
            if globals()[feature].__code__.co_argcount == 2:
                if isinstance(globals()[feature](AAAfreq, noised_signals_freq[p]), (list, tuple, np.ndarray)) == True:
                    result_feature_noise_SNR[feature].append(globals()[feature](AAAfreq, noised_signals_freq[p])[0])
                elif isinstance(globals()[feature](AAAfreq, noised_signals_freq[p]), (int, float, complex)) == True:
                    result_feature_noise_SNR[feature].append(globals()[feature](AAAfreq, noised_signals_freq[p]))
            
            else:
                #ici il faut prendre chaque signal = noise dans le voisinnage et le mettre a la place de range_value
                if isinstance(globals()[feature](noised_signals_real[p]), (list, tuple, np.ndarray)) == True:
                    result_feature_noise_SNR[feature].append(globals()[feature](noised_signals_real[p])[0])
                elif isinstance(globals()[feature](noised_signals_real[p]), (int, float, complex)) == True:
                    result_feature_noise_SNR[feature].append(globals()[feature](noised_signals_real[p]))

        #print(result_feature_noise_SNR[feature])
        #print(len(result_feature_noise_SNR[feature]))
        if np.isnan(coefficient_variation(result_feature_noise_SNR[feature])) == True:
            #print(2)
            noise_CV_SNR[feature].append(0)
        else:
            noise_CV_SNR[feature].append(100 * np.abs(coefficient_variation(result_feature_noise_SNR[feature])))

    plt.figure()
    plt.plot(range_SNR, noise_CV_SNR[feature])
    plt.title('SNR/CV evolution ' + feature)
    plt.xlabel('SNR value (dB)')
    plt.ylabel('Coefficient of variation (%)')


    
    
    
    
    
    
    
#%%


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








#%%


#Final ranking taking into account all algorithms

#To determine this, we normalize each one between 0 and 1 and do the mean of 
#the scores for each feature


#The IQR detects outliers (extreme/absurd values that would strongly bias the normalisation
#otherwise) and takes them off before normalizing. This is useful for RelieF or Chi square
#Where sometimes some frequency feature behave inadequately and have absurd scores that 
#crush all others

#For now we treat them as outliers but it is imperative to look into the reason for 
#those behaviours in future work

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
Final_Ranking = {}
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


for feature in Freq_features + Time_features:
    Final_Ranking_Robustness[feature] = Final_Ranking_Robustness[feature] + Resilient_30dB_1pct[feature]
    #print([Final_Ranking_Robustness[feature], Resilient_30dB_1pct[feature]])



#Now we just need to choose the weight between performance and robustness
#Here we chose a weight of 0.7, favoring performance slightly better

alpha = 0.7

for feature in Freq_features + Time_features:
    Final_Ranking[feature] = alpha * Final_Ranking_feature[feature] + (1 - alpha) * Final_Ranking_Robustness[feature]

Final_Ranking = sorted_result_total = dict(sorted(Final_Ranking.items(), key = lambda item: item[1]))





#%%


#Test to see if we do 20 realisations with different noises at the same SNR
#The variation in the feature value will be big enough to be detected by the 
#algorithms



SNR_value = 20 #dB

noise = np.random.normal(0, sigma_finder_SNR(AAA, 20), AAA.shape[0])

Values = []
for i in range(0, 20):
    Values.append(Zero_crossing_rate(AAA + np.random.normal(0, sigma_finder_SNR(AAA, 20), AAA.shape[0])))

print(coefficient_variation(Values)*100)


#it works


#%%

#This section belongs to the future work section as it is the section aiming
#To decompose the signal into sinuses that have 90% of its energy
#This will be useful for the better way of detecting noise resilience 
#(Cf report or notice)
"""

magnitudes = {}
phases = {}
energy = {}
Total_energy = {}
Cumulative_energy = {}
sorted_energy_indexes = {}
threshold_index = {}
dominant_indices = {}

for key in Calc_FFT.keys():
    magnitudes[key] = np.abs(Calc_FFT[key]) 
    phases[key] = np.angle(Calc_FFT[key]) 
    energy[key] = magnitudes[key] ** 2
    Total_energy[key] = sum(energy[key])
    sorted_energy_indexes[key] = np.argsort(energy[key])[::-1]
    Cumulative_energy[key] = np.cumsum(energy[key])
    threshold_index[key] = np.searchsorted(Cumulative_energy[key], 0.9 * Total_energy[key])
    dominant_indices[key] = sorted_energy_indexes[key][:threshold_index[key] + 1]


    band_width = 3.0 #Hz
    used = np.zeros_like(dominant_indices[key], dtype=bool)
    groups = []
    
    print(1)
    
    for i, idx in enumerate(dominant_indices[key]):
        if used[i]:
            continue
        group = [idx]
        used[i] = True
        for j in range(i + 1, len(dominant_indices[key])):
            if not used[j]:
                if abs(fftFreq[key][dominant_indices[key][j]] - fftFreq[key][idx]) <= band_width:
                    group.append(dominant_indices[key][j])
                    used[j] = True
        groups.append(group)
        
    print(2)
    NN = len(Calc_0[key[5:len(key) - 2]])
    for group in groups:
        group_energy = energy[key][group]
        weights = group_energy/np.sum(group_energy)
        avg_freq = np.sum(positive_freqs[group] * weights)
        avg_amp = np.sum(magnitudes[group]) / (N / 2 * len(group))
        avg_phase = np.sum(phases[group] * weights)
        print(f"{avg_amp:.3f} * cos(2π * {avg_freq:.2f} * t + {avg_phase:.3f})")

"""
"""
#test for one key, here let's take the healthy key N_0
for idx in dominant_indices['FFT (21_BA)']:
    A = magnitudes['FFT (21_BA)'][idx]#/(len(Calc_0['21_BA'])/2) #Normalize amplitude
    f = fftFreq['FFT (21_BA)'][idx]
    phi = phases['FFT (21_BA)'][idx]
    print(f"{A:.3f} * cos(2π * {f:.2f} * t + {phi:.3f})")
    #if magnitudes['FFT (21_BA)'][idx] > 2:
      #  print(magnitudes['FFT (21_BA)'][idx])


#CV_SNR_curves()
"""

#%%

#In this section is one of the first way of comparing the resilience of this feature
#The goal of those tests was to compare the resilience tests between the 
#Coefficient of variation method and the Pearson correlation method (2 different
#methods to quantify how the features resist to the measurment defaults)

#After those tests, we found quite similar results and decided to go with the 
#Coefficient of variation method as it was the most used in literature

"""
# A = Calc_10[0]['N_0_0'].ravel()
A0 = np.linspace(0, 100, 1000)
A = np.sin(A0)

t = np.arange(0, len(A), 1)
sigma_float = np.arange(0.2, 1.9, 0.4)
sigma = [round(k,1) for k in sigma_float]
sigma = [0] + sigma
noise = {}
noised_signal = {}
for i in range(0,len(sigma)):
    noise[sigma[i]] = np.random.normal(0, sigma[i], A.shape[0])
    noised_signal[sigma[i]] = A + noise[sigma[i]]
    
drift_values = np.linspace(0, 1, 13)
drift_signals = {}
offset_values = np.linspace(0, 3, 13)
offset_signals = {}
for i in range(0, len(drift_values)):
    drift_signals[drift_values[i]] = A + drift_values[i]*t/200
    offset_signals[offset_values[i]] = A + offset_values[i]
    
    



# plt.figure()
# plt.plot(noise[0.2])


# plt.figure()
# plt.plot(noise[0.6])

features = {}
result_feature_noise = {}
result_feature_drift = {}
result_feature_offset = {}
for i in Time_features:
    result_feature_noise[i] = []
    result_feature_drift[i] = []
    result_feature_offset[i] = []
    print(i)
    for key in noised_signal.keys():

        #print(isinstance(globals()[i](noised_signal[key]), (int, float, complex)))
        #print(isinstance(globals()[i](noised_signal[key]), (list, tuple, np.ndarray)))

        
        if isinstance(globals()[i](noised_signal[key]), (list, tuple, np.ndarray)) == True:
            result_feature_noise[i].append(globals()[i](noised_signal[key])[0])
        elif isinstance(globals()[i](noised_signal[key]), (int, float, complex)) == True:
            result_feature_noise[i].append(globals()[i](noised_signal[key]))
        else:
            print('Error in the function output')
            
    
    for key in drift_signals.keys():

        #print(isinstance(globals()[i](noised_signal[key]), (int, float, complex)))
        #print(isinstance(globals()[i](noised_signal[key]), (list, tuple, np.ndarray)))

        
        if isinstance(globals()[i](drift_signals[key]), (list, tuple, np.ndarray)) == True:
            result_feature_drift[i].append(globals()[i](drift_signals[key])[0])
        elif isinstance(globals()[i](drift_signals[key]), (int, float, complex)) == True:
            result_feature_drift[i].append(globals()[i](drift_signals[key]))
        else:
            print('Error in the function output')
            
            
            
    for key in offset_signals.keys():

        #print(isinstance(globals()[i](noised_signal[key]), (int, float, complex)))
        #print(isinstance(globals()[i](noised_signal[key]), (list, tuple, np.ndarray)))

        
        if isinstance(globals()[i](offset_signals[key]), (list, tuple, np.ndarray)) == True:
            result_feature_offset[i].append(globals()[i](offset_signals[key])[0])
        elif isinstance(globals()[i](offset_signals[key]), (int, float, complex)) == True:
            result_feature_offset[i].append(globals()[i](offset_signals[key]))
        else:
            print('Error in the function output')
            





noise_Pearson = {}
drift_Pearson = {}
offset_Pearson = {}
for i in Time_features:
    A = [pearsonr(result_feature_noise[i], sigma)[0], pearsonr(result_feature_noise[i], sigma)[1]]
    noise_Pearson[i] = np.abs(A[0])\
        
    
    B = [pearsonr(result_feature_drift[i], drift_values)[0], pearsonr(result_feature_drift[i], drift_values)[1]]
    drift_Pearson[i] = np.abs(B[0])
    
    
    C = [pearsonr(result_feature_offset[i], offset_values)[0], pearsonr(result_feature_offset[i], offset_values)[1]]
    offset_Pearson[i] = np.abs(C[0])
    
    
    
sorted_noise_Pearson = dict(sorted(noise_Pearson.items(), key = lambda item: item[1]))    
sorted_drift_Pearson = dict(sorted(drift_Pearson.items(), key = lambda item: item[1]))
sorted_offset_Pearson = dict(sorted(offset_Pearson.items(), key = lambda item: item[1]))

"""

"""
    

normalized_features_noise = {k: normalisation(v, [0,1]) for k,v in result_feature_noise.items()}
normalized_features_drift = {k: normalisation(v, [0,1]) for k,v in result_feature_drift.items()}
normalized_features_offset = {k: normalisation(v, [0,1]) for k,v in result_feature_offset.items()}



noise_CV = {}
noise_CV_2 = {}
drift_CV = {}
drift_CV_2 = {}
offset_CV = {}
offset_CV_2 = {}

for i in Time_features:
    noise_CV[i] = coefficient_variation(normalized_features_noise[i])
    noise_CV_2[i] = coefficient_variation(result_feature_noise[i])
    drift_CV[i] = coefficient_variation(normalized_features_drift[i])
    drift_CV_2[i] = coefficient_variation(result_feature_drift[i])
    offset_CV[i] = coefficient_variation(normalized_features_offset[i])
    offset_CV_2[i] = coefficient_variation(result_feature_offset[i])


#We filter the values so that the extreme values won't impend the lecture of the graph
sorted_noise_CV = dict(sorted(noise_CV.items(), key = lambda item: item[1]))
sorted_drift_CV = dict(sorted(drift_CV.items(), key = lambda item: item[1]))
sorted_offset_CV = dict(sorted(offset_CV.items(), key = lambda item: item[1]))

sorted_noise_CV_2 = dict(sorted(noise_CV_2.items(), key = lambda item: item[1]))
sorted_drift_CV_2 = dict(sorted(drift_CV_2.items(), key = lambda item: item[1]))
sorted_offset_CV_2 = dict(sorted(offset_CV_2.items(), key = lambda item: item[1]))
#filtered_noise_CV = {k: v for k,v in sorted_noise_CV.items() if np.abs(v)<=1}
#extreme_noise_CV = {k: v for k,v in sorted_noise_CV.items() if np.abs(v)>1}


#This normalises a list of function amplitudes into a range of values, plage = [a,b]







Time_features_noise_CV = sorted_noise_CV.keys()

plt.figure(figsize=(10, 6))
bars = plt.barh(Time_features_noise_CV, sorted_noise_CV.values(), color='skyblue')
plt.xlabel('Noise resilience CV')


Time_features_noise_Pearson = sorted_noise_Pearson.keys()

plt.figure(figsize=(10, 6))
bars = plt.barh(Time_features_noise_Pearson, sorted_noise_Pearson.values(), color='skyblue')
plt.xlabel('Noise resilience Pearson')





Time_features_drift_CV = sorted_drift_CV.keys()

plt.figure(figsize=(10, 6))
bars = plt.barh(Time_features_drift_CV, sorted_drift_CV.values(), color='skyblue')
plt.xlabel('Drift resilience CV')


Time_features_drift_Pearson = sorted_noise_Pearson.keys()

plt.figure(figsize=(10, 6))
bars = plt.barh(Time_features_drift_Pearson, sorted_drift_Pearson.values(), color='skyblue')
plt.xlabel('Drift resilience Pearson')







Time_features_offset_CV = sorted_offset_CV.keys()

plt.figure(figsize=(10, 6))
bars = plt.barh(Time_features_offset_CV, sorted_offset_CV.values(), color='skyblue')
plt.xlabel('Offset resilience CV')


Time_features_offset_Pearson = sorted_offset_Pearson.keys()

plt.figure(figsize=(10, 6))
bars = plt.barh(Time_features_offset_Pearson, sorted_offset_Pearson.values(), color='skyblue')
plt.xlabel('Offset resilience Pearson')
"""



#%%


#In this section you'll find the old way of looking for feature resilience
#with the 5 different main magnitudes (cf Python notice)
#This way can still be worthwile to look into as we still don't have any ways
#Like the SNR to know a good range to look into, so yeah maybe it could be nice
#To have this.


"""
#Tests



# AAA = Calc_10[0]['N_0_0'].ravel()
A0 = np.linspace(0, 100, 1000)
AAA = np.sin(A0)
orders = [0.01, 0.1, 1, 10, 100]

for q in orders:
    print(q)
    t = np.linspace(0, 100, 1000)
    sigma = np.linspace(q, q*2 , 10)
    noise = {}
    noised_signal = {}
    for i in range(0,len(sigma)):
        noise[sigma[i]] = np.random.normal(0, sigma[i], AAA.shape[0])
        noised_signal[sigma[i]] = AAA + noise[sigma[i]]
        
    drift_values = np.linspace(q, q*2, 10)
    drift_signals = {}
    offset_values = np.linspace(q, q*2, 10)
    offset_signals = {}
    sine_values = np.linspace(q, q*2, 10)
    sine_signals = {}
    for i in range(0, len(drift_values)):
        drift_signals[drift_values[i]] = AAA + drift_values[i]*t + noise[sigma[i]]
        offset_signals[offset_values[i]] = AAA + offset_values[i] 
        freqs = sine_values[:i+1]
        sine_signals[sine_values[i]] = sum(np.sin(2*np.pi*f*t) for f in freqs) + AAA
        
    

    features = {}
    result_feature_noise = {}
    result_feature_drift = {}
    result_feature_offset = {}
    result_feature_sine = {}
    for i in Time_features:
        result_feature_noise[i] = []
        result_feature_drift[i] = []
        result_feature_offset[i] = []
        result_feature_sine[i] = []
        for key in noised_signal.keys():

            #print(isinstance(globals()[i](noised_signal[key]), (int, float, complex)))
            #print(isinstance(globals()[i](noised_signal[key]), (list, tuple, np.ndarray)))

            
            if isinstance(globals()[i](noised_signal[key]), (list, tuple, np.ndarray)) == True:
                result_feature_noise[i].append(globals()[i](noised_signal[key])[0])
            elif isinstance(globals()[i](noised_signal[key]), (int, float, complex)) == True:
                result_feature_noise[i].append(globals()[i](noised_signal[key]))
            else:
                print('Error in the function output')
                
        
        for key in drift_signals.keys():

            #print(isinstance(globals()[i](noised_signal[key]), (int, float, complex)))
            #print(isinstance(globals()[i](noised_signal[key]), (list, tuple, np.ndarray)))

            
            if isinstance(globals()[i](drift_signals[key]), (list, tuple, np.ndarray)) == True:
                result_feature_drift[i].append(globals()[i](drift_signals[key])[0])
            elif isinstance(globals()[i](drift_signals[key]), (int, float, complex)) == True:
                result_feature_drift[i].append(globals()[i](drift_signals[key]))
            else:
                print('Error in the function output')
                
                
                
        for key in offset_signals.keys():

            #print(isinstance(globals()[i](noised_signal[key]), (int, float, complex)))
            #print(isinstance(globals()[i](noised_signal[key]), (list, tuple, np.ndarray)))

            
            if isinstance(globals()[i](offset_signals[key]), (list, tuple, np.ndarray)) == True:
                result_feature_offset[i].append(globals()[i](offset_signals[key])[0])
            elif isinstance(globals()[i](offset_signals[key]), (int, float, complex)) == True:
                result_feature_offset[i].append(globals()[i](offset_signals[key]))
            else:
                print('Error in the function output')
                
        for key in sine_signals.keys():

            #print(isinstance(globals()[i](noised_signal[key]), (int, float, complex)))
            #print(isinstance(globals()[i](noised_signal[key]), (list, tuple, np.ndarray)))

            
            if isinstance(globals()[i](sine_signals[key]), (list, tuple, np.ndarray)) == True:
                result_feature_sine[i].append(globals()[i](sine_signals[key])[0])
            elif isinstance(globals()[i](sine_signals[key]), (int, float, complex)) == True:
                result_feature_sine[i].append(globals()[i](sine_signals[key]))
            else:
                print('Error in the function output')
                

    """
    
"""
    noise_Pearson = {}
    drift_Pearson = {}
    offset_Pearson = {}
    sine_Pearson = {}
    
    for i in Time_features:
        if np.isnan(pearsonr(result_feature_noise[i], sigma))[0] == True:
            noise_Pearson[i] == 0
        else:
            A = [pearsonr(result_feature_noise[i], sigma)[0], pearsonr(result_feature_noise[i], sigma)[1]]
            noise_Pearson[i] = np.abs(A[0])\
            
        if np.isnan(pearsonr(result_feature_drift[i], drift_values))[0] == True:
            drift_Pearson[i] = 0
        else:
            B = [pearsonr(result_feature_drift[i], drift_values)[0], pearsonr(result_feature_drift[i], drift_values)[1]]
            drift_Pearson[i] = np.abs(B[0])
        
        if np.isnan(pearsonr(result_feature_offset[i], offset_values))[0] == True:
            offset_Pearson[i] = 0
        else: 
            C = [pearsonr(result_feature_offset[i], offset_values)[0], pearsonr(result_feature_offset[i], offset_values)[1]]
            offset_Pearson[i] = np.abs(C[0])
            
            
        if np.isnan(pearsonr(result_feature_sine[i], sine_values))[0] == True:
            sine_Pearson[i] = 0
        else: 
            D = [pearsonr(result_feature_sine[i], sine_values)[0], pearsonr(result_feature_sine[i], sine_values)[1]]
            sine_Pearson[i] = np.abs(D[0])
        
        
    print(noise_Pearson.keys())
    sorted_noise_Pearson = dict(sorted(noise_Pearson.items(), key = lambda item: item[1]))    
    sorted_drift_Pearson = dict(sorted(drift_Pearson.items(), key = lambda item: item[1]))
    sorted_offset_Pearson = dict(sorted(offset_Pearson.items(), key = lambda item: item[1]))
    sorted_sine_Pearson = dict(sorted(sine_Pearson.items(), key = lambda item: item[1]))


    """
"""

    # normalized_features_noise = {k: normalisation(v, [0,1]) for k,v in result_feature_noise.items()}
    # normalized_features_drift = {k: normalisation(v, [0,1]) for k,v in result_feature_drift.items()}
    # normalized_features_offset = {k: normalisation(v, [0,1]) for k,v in result_feature_offset.items()}
    # normalized_features_sine = {k: normalisation(v, [0,1]) for k,v in result_feature_sine.items()}



    noise_CV = {}
    drift_CV = {}
    offset_CV = {}
    sine_CV = {}
    for i in Time_features:
        if np.isnan(coefficient_variation(result_feature_noise[i])) == True:
            noise_CV[i] = 0
        else:
            noise_CV[i] = np.abs(coefficient_variation(result_feature_noise[i]))
            
        if np.isnan(coefficient_variation(result_feature_drift[i])) == True:
            drift_CV[i] = 0
        else:
            drift_CV[i] = np.abs(coefficient_variation(result_feature_drift[i]))
            
        if np.isnan(coefficient_variation(result_feature_offset[i])) == True:
            offset_CV[i] = 0
        else:
            offset_CV[i] = np.abs(coefficient_variation(result_feature_offset[i]))
            
        if np.isnan(coefficient_variation(result_feature_sine[i])) == True:
            sine_CV[i] = 0
        else:
            sine_CV[i] = np.abs(coefficient_variation(result_feature_sine[i]))



    sorted_noise_CV = dict(sorted(noise_CV.items(), key = lambda item: item[1]))
    sorted_drift_CV = dict(sorted(drift_CV.items(), key = lambda item: item[1]))
    sorted_offset_CV = dict(sorted(offset_CV.items(), key = lambda item: item[1]))
    sorted_sine_CV = dict(sorted(sine_CV.items(), key = lambda item: item[1]))



    Time_features_noise_CV = sorted_noise_CV.keys()
    count_filter = {}
    plt.figure(figsize=(10, 6))
    
    bars = plt.barh([v for v in sorted_noise_CV.keys() if sorted_noise_CV[v] < 0.1], [k for k in sorted_noise_CV.values() if k<0.1], color='skyblue')
    plt.xlabel(f'Noise resilience CV magnitude {q}')


    # Time_features_noise_Pearson = sorted_noise_Pearson.keys()

    # plt.figure(figsize=(10, 6))
    # bars = plt.barh(Time_features_noise_Pearson, sorted_noise_Pearson.values(), color='skyblue')
    # plt.xlabel(f'Noise resilience Pearson magnitude{q}')



    Time_features_drift_CV = sorted_drift_CV.keys()
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh([v for v in sorted_drift_CV.keys() if sorted_drift_CV[v] < 0.1], [k for k in sorted_drift_CV.values() if k<0.1], color='skyblue')
    plt.xlabel(f'Drift resilience CV magnitude{q}')
    
    
    
    
    
    # Time_features_drift_Pearson = sorted_noise_Pearson.keys()
    
    # plt.figure(figsize=(10, 6))
    # bars = plt.barh(Time_features_drift_Pearson, sorted_drift_Pearson.values(), color='skyblue')
    # plt.xlabel(f'Drift resilience Pearson magnitude{q}')



    Time_features_offset_CV = sorted_offset_CV.keys()

    plt.figure(figsize=(10, 6))
    bars = plt.barh([v for v in sorted_offset_CV.keys() if sorted_offset_CV[v] < 0.1], [k for k in sorted_offset_CV.values() if k<0.1], color='skyblue')
    plt.xlabel(f'Offset resilience CV magnitude{q}')


    # Time_features_offset_Pearson = sorted_offset_Pearson.keys()

    # plt.figure(figsize=(10, 6))
    # bars = plt.barh(Time_features_offset_Pearson, sorted_offset_Pearson.values(), color='skyblue')
    # plt.xlabel(f'Offset resilience Pearson magnitude{q}')


    Time_features_sine_CV = sorted_sine_CV.keys()

    plt.figure(figsize=(10, 6))
    bars = plt.barh([v for v in sorted_sine_CV.keys() if sorted_sine_CV[v] < 0.1], [k for k in sorted_sine_CV.values() if k<0.1], color='skyblue')
    plt.xlabel(f'Sine resilience CV magnitude{q}')



    # Time_features_sine_Pearson = sorted_sine_Pearson.keys()

    # plt.figure(figsize=(10, 6))
    # bars = plt.barh(Time_features_sine_Pearson, sorted_sine_Pearson.values(), color='skyblue')
    # plt.xlabel(f'Sine resilience Pearson magnitude{q}')

"""




#%%



#Here lies the old method of calculating the CV (with the +- 2% value range of 
#SNR to artificlaly create a value list for the calculation of the CV)





"""


#We've taken a range from 10 db (worst performance allowed) to 40 db (highest performance 
#considered, above that the SNR 'll be so high/good that noise resiliency doesn't matter).
range_SNR = np.linspace(10,40,155)

#Then we find the sigma associated to this for our signal and create the noise
#associated to this sigma

#Then we look at the value of the CV
#Because for the calculation of the CV a range of values is needed but here we're looking
#to know the CV value for each point, for each point we'll take a range from 
#Value - Value/100*2 to Value + Value/100*2 with 41 points to try and keep 
#the neighborhood of the point as accurate as possible


noised_signals = {}
#drifted_signals = {}
sigma = {}
CV_values = {}
result_feature_noise_SNR = {}
noise_CV_SNR = {}
#drift_CV_SNR = {}

for i in Time_features:
    noise_CV_SNR[i] = []
    #drift_CV_SNR[i] = []

for feature in Time_features:
    print(feature)
    for i in range(0, len(range_SNR)):
        sigma[i] = sigma_finder_SNR(AAA, range_SNR[i])
        noised_signals[i] = []
        #drifted_signals[i] = []
        
        range_value = []
        range_value = np.linspace(sigma[i] - sigma[i]/1000*20, sigma[i] + sigma[i]/1000*20, 41)
        
        for range_sigma in range_value:
            noised_signals[i].append(AAA + np.random.normal(0, range_sigma, AAA.shape[0]))
    
        
        
        #print(i)
        result_feature_noise_SNR[feature] = []
        #print(globals()[i](range_value))
        #print(j)
        for signal in noised_signals[i]:
            #Here we must take each signal = noise in the neighborhood and put it instead of range_value
            if isinstance(globals()[feature](signal), (list, tuple, np.ndarray)) == True:
                result_feature_noise_SNR[feature].append(globals()[feature](signal)[0])
            elif isinstance(globals()[feature](signal), (int, float, complex)) == True:
                result_feature_noise_SNR[feature].append(globals()[feature](signal))

        #print(result_feature_noise_SNR[feature])
        #print(len(result_feature_noise_SNR[feature]))
        if np.isnan(coefficient_variation(result_feature_noise_SNR[feature])) == True:
            #print(2)
            noise_CV_SNR[feature].append(0)
        else:
            noise_CV_SNR[feature].append(100 * np.abs(coefficient_variation(result_feature_noise_SNR[feature])))



"""



#%%

"""
#Here lies the previous tests done to check if there was a coherence in the 
#features found with the magnitude method and the CV method

magnitude_0_01 = ['Min', 'hist_lower_bound', 'Max', 'hist_higher_bound', 'latitude_factor', 'Margin_Index',
                  'Clearance', 'Peak_to_peak', 'Peak_Val', 'EO', 'IF', 'Crest', 'Log_Log_Ratio',
                  'Kurtosis', 'Wilson_amplitude', 'Energy', 'Variance', 'Mean_abs_modif2',
                  'RMS', 'STD', 'Mean_abs_modif1', 'Mean_abs', 'SRAV', 'SF', 'SDIF', 'NNNL']


SNR_magnitude_0_01 = []

for feature in Time_features:
    if noise_CV_SNR[feature][17] <=10: 
        SNR_magnitude_0_01.append(feature)
        
        
for i in SNR_magnitude_0_01:
    if i not in magnitude_0_01:
        print(i)


print(3)



for j in magnitude_0_01:
    if j not in SNR_magnitude_0_01:
        print(j)

"""


#%%


#This section contains a test to know which features overlap on the rankings
#This acted as some sort of pre-performance ranking

"""

#filters a dictionnary d with the n bigger values 
def filter_dictionnary(d, n):
    
    top_items = sorted(d.items(), key=lambda item: item[1], reverse=True)[:n]
    return dict(top_items)


Healthy_Info_10 = filter_dictionnary(Healthy_Info_10, 15)

Healthy_Pearson_10 = filter_dictionnary(Healthy_Pearson_10, 15)

Healthy_RelieF_10 = filter_dictionnary(Healthy_RelieF_10, 15)

Healthy_Chi_10 = filter_dictionnary(Healthy_Chi_10, 15)

Ranking = {}


for feature in Time_features:
    if feature in Healthy_Info_5.keys():
        if feature in Ranking.keys():
            Ranking[feature] = Ranking[feature] + 1
        else:
            Ranking[feature] = 1
    
    if feature in Healthy_Pearson_10.keys():
        if feature in Ranking.keys():
            Ranking[feature] = Ranking[feature] + 1
        else:
            Ranking[feature] = 1
            
    if feature in Healthy_RelieF_10.keys():
        if feature in Ranking.keys():
            Ranking[feature] = Ranking[feature] + 1
        else:
            Ranking[feature] = 1
            
    if feature in Healthy_Chi_10.keys():
        if feature in Ranking.keys():
            Ranking[feature] = Ranking[feature] + 1
        else:
            Ranking[feature] = 1


Ranking = sorted(Ranking.items(), key=lambda item: item[1], reverse=True)

"""