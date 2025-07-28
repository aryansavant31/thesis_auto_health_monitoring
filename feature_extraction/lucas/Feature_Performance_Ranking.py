# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 16:50:10 2025

@author: lfernan4
"""


import scipy.io
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.stats import entropy
from Data_pre_processing import *
from Time_features import *
from Frequency_features import *
from Ranking_algos import *
import inspect
import ast

#First let's list all the features in a dictionary




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



#Test to artificially create more datasets

#Calc_0['N_0_2'] = Calc_0['N_0'][0:np.shape(Calc_0['N_0'])[0]//2]
#Calc_0['N_0'] = Calc_0['N_0'][np.shape(Calc_0['N_0'])[0]//2:]


#This function only segments the healthy segment, seeing as we only got one


def segmenting_healthy(Calc_0, number):
    Calc_2 = {}
    Calc_FFT_2 = {}
    A = {}
    fftFreq = {}
    spacing = 1/48000
    for i in range(0, number):
        Calc_2[f"N_0_{i}"] = Calc_0['N_0'][i*np.shape(Calc_0['N_0'])[0]//number:(i+1)*np.shape(Calc_0['N_0'])[0]//number]
        A[f"N_0_{i}"] = fft(Calc_2[f"N_0_{i}"])
        Calc_FFT_2['FFT (' + f"N_0_{i}" + ')'] = np.zeros(len(A[f"N_0_{i}"]))
        for j in range(0, len(A[f"N_0_{i}"])):
            Calc_FFT_2['FFT (' + f"N_0_{i}" + ')'][i] = np.sqrt( np.real(A[f"N_0_{i}"][i])**2 + np.imag(A[f"N_0_{i}"][i])**2)
        fftFreq['FFT (' + f"N_0_{i}" + ')'] = np.fft.rfftfreq(len(Calc_FFT_2['FFT (' + f"N_0_{i}" + ')']), spacing)
        
        #This line here is to get rid of the symetric part in the negative real axis
        #and to have fftFreq and Calc_FFT of the same length for more efficient operations
        Calc_FFT_2['FFT (' + f"N_0_{i}" + ')'] = Calc_FFT_2['FFT (' + f"N_0_{i}" + ')'][:len(Calc_FFT_2['FFT (' + f"N_0_{i}" + ')'])//2 +1]
    
    
    for key in Calc_0.keys():
        if key != 'N_0':
            Calc_2[key] = Calc_0[key]
            A[key] = fft(Calc_0[key])
            Calc_FFT_2['FFT (' + key + ')'] = np.zeros(len(A[key]))
            for i in range(0, len(A[key])):
                Calc_FFT_2['FFT (' + key + ')'][i] = np.sqrt( np.real(A[key][i])**2 + np.imag(A[key][i])**2)
            fftFreq['FFT (' + key + ')'] = np.fft.rfftfreq(len(Calc_FFT_2['FFT (' + key + ')']), spacing)
            
            #This line here is to get rid of the symetric part in the negative real axis
            #and to have fftFreq and Calc_FFT of the same length for more efficient operations
            Calc_FFT_2['FFT (' + key + ')'] = Calc_FFT_2['FFT (' + key + ')'][:len(Calc_FFT_2['FFT (' + key + ')'])//2 +1]
            
    return(Calc_2, fftFreq, Calc_FFT_2)









#Problem of time execution with EO
def time_comparison_info_gain(Calc_0, bin_number):
    result_time = {}
    
    
    for i in Time_features:
        result_time[i] = information_gain(Calc_0, globals()[i], bin_number)
        print(i)
        
        
    sorted_result_time = dict(sorted(result_time.items(), key = lambda item: item[1]))
    
    return(sorted_result_time)
    

#We have a second comparison for the frequency features as we need to 
#Take the FFT dictionary into input and there can also be some 2 argument features
#Unlike time features

    
def freq_comparison_info_gain(fftFreq, Calc_FFT, bin_number):
    result_freq = {}
    
    for i in Freq_features:
        print(i)
        if globals()[i].__code__.co_argcount == 2:
            result_freq[i] = information_gain_freq(Calc_FFT, fftFreq, globals()[i], bin_number)
        else:
            result_freq[i] = information_gain(Calc_FFT, globals()[i], bin_number)
        
        
        
    sorted_result_freq = dict(sorted(result_freq.items(), key = lambda item: item[1]))
        
        
        
        
        
    sorted_result_freq = dict(sorted(result_freq.items(), key = lambda item: item[1]))
    
    return(sorted_result_freq)






def total_comparison_info_gain(Calc_0, bin_number):
    result_total = {}
    
    for i in Time_features:
        result_total[i] = information_gain(Calc_0, globals()[i], bin_number)
        
    for i in Freq_features:
        result_total[i] = information_gain(Calc_0, globals()[i], bin_number)
    
    sorted_result_total = dict(sorted(result_total.items(), key = lambda item: item[1]))
    
    
    
    
def time_comparison_Chi_Square(Calc_0, bin_number):
    result_time = {}
    
    
    for i in Time_features:
        result_time[i] = Chi_Square(Calc_0, globals()[i], bin_number)
        print(i)
        
    #We rank the values according to X^2
    sorted_result_time = dict(sorted(result_time.items(), key = lambda item: item[1][0]))
    
    return(sorted_result_time)

    

def freq_comparison_Chi_Square(fftFreq, Calc_FFT, bin_number):
    result_freq = {}
    
    for i in Freq_features:
        print(i)
        if globals()[i].__code__.co_argcount == 2:
            result_freq[i] = Chi_Square_freq(Calc_FFT, fftFreq, globals()[i], bin_number)
        else:
            result_freq[i] = Chi_Square(Calc_FFT, globals()[i], bin_number)
        
        
        
    sorted_result_freq = dict(sorted(result_freq.items(), key = lambda item: item[1]))
        
        

    
    return(sorted_result_freq)    

    
    
    
def time_comparison_Pearson(Calc_0):
    result_time = {}
    
    
    for i in Time_features:
        print(i)
        result_time[i] = Pearson_Corr(Calc_0, globals()[i])
        result_time[i][0] = np.abs(result_time[i][0])
        
        
    
    sorted_result_time = dict(sorted(result_time.items(), key = lambda item: item[1]))
    
    return(sorted_result_time)    
    


#Probleme avec la fonction Pearson_Corr_Freq
    
    
def freq_comparison_Pearson(fftFreq, Calc_FFT):
    result_freq = {}
    
    
    for i in Freq_features:
        #print(inspect.signature(i))
        print(i)
        if globals()[i].__code__.co_argcount == 2:
            result_freq[i] = Pearson_Corr_freq(Calc_FFT, fftFreq, globals()[i])
            result_freq[i][0] = np.abs(result_freq[i][0])
        else:
            result_freq[i] = Pearson_Corr(Calc_FFT, globals()[i])
            result_freq[i][0] = np.abs(result_freq[i][0])
        
        
    #We sort according to the Pearson correlation coeff, the probabilty needs
    #be checked by hand
    
    
    sorted_result_freq = dict(sorted(result_freq.items(), key = lambda item: item[1][0]))
    
    return(sorted_result_freq)     


    


def time_comparison_RelieF(Calc_0):
    result_time = {}
    
    
    for i in Time_features:
        result_time[i] = RelieF(Calc_0, Time_features, globals()[i])
        print(i)

    
    sorted_result_time = dict(sorted(result_time.items(), key = lambda item: item[1]))
    
    
    return(sorted_result_time)



def freq_comparison_RelieF(fftFreq, Calc_FFT):
    result_freq = {}
    
    
    for i in Freq_features:
        #print(inspect.signature(i))
        print(i)
        if globals()[i].__code__.co_argcount == 2:
            result_freq[i] = RelieF_freq(Calc_FFT, fftFreq, Freq_features, globals()[i])
            #result_freq[i][0] = np.abs(result_freq[i][0])
        else:
            result_freq[i] = RelieF(Calc_FFT, Freq_features, globals()[i])
            #result_freq[i][0] = np.abs(result_freq[i][0])
        
        
    #We sort according to the Pearson correlation coeff, the probabilty needs
    #be checked by hand
    
    
    sorted_result_freq = dict(sorted(result_freq.items(), key = lambda item: item[1]))
    
    
    
    return(sorted_result_freq)  
    
#%%







#%%
#Comparaison de la segmentation healthy




#Calc_3 = segmenting_healthy(Calc_0, 3)
#Calc_5 = segmenting_healthy(Calc_0, 5)
Calc_10 = segmenting_healthy(Calc_0, 10)
np.save('Calc_10.npy', Calc_10)

#%%

Healthy_Chi_10 = time_comparison_Chi_Square(Calc_10[0], 3)
Healthy_Pearson_10 = time_comparison_Pearson(Calc_10[0])
Healthy_RelieF_10 = time_comparison_RelieF(Calc_10[0])
Healthy_Info_10 = time_comparison_info_gain(Calc_10[0], 3)

#%%
healthy_freq_Chi = freq_comparison_Chi_Square(Calc_10[1], Calc_10[2], 3)
healthy_freq_RelieF = freq_comparison_RelieF(Calc_10[1], Calc_10[2])
healthy_freq_Pearson = freq_comparison_Pearson(Calc_10[1], Calc_10[2])

healthy_freq_info = freq_comparison_info_gain(Calc_10[1], Calc_10[2], 3)


#%%


#Healthy_Info = time_comparison_info_gain(Calc_0, 3)
#Healthy_Info_3 = time_comparison_info_gain(Calc_3[0], 3)
#Healthy_Info_5 = time_comparison_info_gain(Calc_5[0], 3)



#%%






Freq_features_Chi = list(healthy_freq_Chi.keys())
Freq_values_Chi = []
for i in healthy_freq_Chi.values():
    Freq_values_Chi.append(i[0])


np.save('Freq_values_Chi.npy', Freq_values_Chi)
np.save('Freq_features_Chi.npy', Freq_features_Chi)


plt.figure(figsize=(10, 6))
bars = plt.barh(Freq_features_Chi, Freq_values_Chi, color='skyblue')
#bars = plt.barh(Time_features_RelieF, Time_values_RelieF, color='skyblue')

plt.xlabel('Freq X^2 value')    


Freq_features_RelieF = list(healthy_freq_RelieF.keys())
Freq_values_RelieF = []
for i in healthy_freq_RelieF.values():
        Freq_values_RelieF.append(i) 

np.save('Freq_values_RelieF.npy', Freq_values_RelieF)
np.save('Freq_features_RelieF.npy', Freq_features_RelieF)

plt.figure(figsize=(10, 6))
bars = plt.barh([v for v in healthy_freq_RelieF.keys() if np.abs(healthy_freq_RelieF[v]) < 200000], [k for k in healthy_freq_RelieF.values() if np.abs(k)<200000], color = 'skyblue')
#bars = plt.barh(Time_features_RelieF, Time_values_RelieF, color='skyblue')

plt.xlabel('RelieF weights freq')


Freq_features_info = list(healthy_freq_info.keys())
Freq_values_info = []
for i in healthy_freq_info.values():
    Freq_values_info.append(i)  


np.save('Freq_values_info.npy', Freq_values_info)
np.save('Freq_features_info.npy', Freq_features_info)


plt.figure(figsize=(10, 6))
bars = plt.barh(Freq_features_info, Freq_values_info, color='skyblue')
plt.xlabel('Information gain')




Freq_features_Pearson = list(healthy_freq_Pearson.keys())
Freq_values_Pearson = []
for i in healthy_freq_Pearson.values():
    Freq_values_Pearson.append(i[0])#wE TAKE THE CORRELATION value
    
np.save('Freq_values_Pearson.npy', Freq_values_Pearson)
np.save('Freq_features_Pearson.npy', Freq_features_Pearson)

plt.figure(figsize=(10, 6))
colors = ['red' if i[1]>0.05 else 'skyblue' for i in healthy_freq_Pearson.values()]
bars = plt.barh(Freq_features_Pearson, Freq_values_Pearson, color=colors)
plt.xlabel('Pearson Coefficient freq') 



#%%
Time_features_Pearson = list(Healthy_Pearson_10.keys())
Time_values_Pearson = []
for i in Healthy_Pearson_10.values():
    Time_values_Pearson.append(i[0])  #wE TAKE THE CORRELATION value

np.save('Time_values_Pearson.npy', Time_values_Pearson)
np.save('Time_features_Pearson.npy', Time_features_Pearson)

plt.figure(figsize=(10, 6))
bars = plt.barh(Time_features_Pearson, Time_values_Pearson, color='skyblue')
plt.xlabel('Pearson Coefficient')



Time_features_Chi = list(Healthy_Chi_10.keys())
Time_values_Chi = []
for i in Healthy_Chi_10.values():
    Time_values_Chi.append(i[0])  #wE TAKE THE CORRELATION value

np.save('Time_values_Chi.npy', Time_values_Chi)
np.save('Time_features_Chi.npy', Time_features_Chi)

colors = ['red' if i[1]>0.05 else 'skyblue' for i in Healthy_Chi_10.values()]
plt.figure(figsize=(10, 6))
bars = plt.barh(Time_features_Chi, Time_values_Chi, color=colors)
plt.xlabel('X^2 value')



Time_features_Info_Gain = list(Healthy_Info_10.keys())
Time_values_Info_Gain = []
for i in Healthy_Info_10.values():
    Time_values_Info_Gain.append(i) 

np.save('Time_values_Info_Gain.npy', Time_values_Info_Gain)
np.save('Time_features_Info_Gain.npy', Time_features_Info_Gain)

plt.figure(figsize=(10, 6))
bars = plt.barh(Time_features_Info_Gain, Time_values_Info_Gain, color='skyblue')
plt.xlabel('Information Gain')



Time_features_RelieF = list(Healthy_RelieF_10.keys())
Time_values_RelieF = []
for i in Healthy_RelieF_10.values():
    if np.abs(i) < 20:
        Time_values_RelieF.append(i) 

np.save('Time_values_RelieF.npy', Time_values_RelieF)
np.save('Time_features_RelieF.npy', Time_features_RelieF)


plt.figure(figsize=(10, 6))
bars = plt.barh([v for v in Healthy_RelieF_10.keys() if np.abs(Healthy_RelieF_10[v]) < 20], [k for k in Healthy_RelieF_10.values() if np.abs(k)<20], color = 'skyblue')
#bars = plt.barh(Time_features_RelieF, Time_values_RelieF, color='skyblue')

plt.xlabel('RelieF weights')


# Time_features_Info_Gain_5 = Healthy_Info_5.keys()
# Time_values_Info_Gain_5 = []
# for i in Healthy_Info_5.values():
#     Time_values_Info_Gain_5.append(i) 


# plt.figure(figsize=(10, 6))
# bars = plt.barh(Time_features_Info_Gain_5, Time_values_Info_Gain_5, color='skyblue')
# plt.xlabel('Information Gain 5 sections')




# Time_features_Info_Gain_3 = Healthy_Info_3.keys()
# Time_values_Info_Gain_3 = []
# for i in Healthy_Info_3.values():
#     Time_values_Info_Gain_3.append(i) 


# plt.figure(figsize=(10, 6))
# bars = plt.barh(Time_features_Info_Gain_3, Time_values_Info_Gain_3, color='skyblue')
# plt.xlabel('Information Gain 3 sections')





#%%




    

