# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 13:25:25 2025

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
from scipy.stats import chi2
from scipy.stats import pearsonr
#import sklearn





#%%
#########################################

#Simple test information algorithm (The real ranking algo functions begin next section)
#This section contains the frequency binning function

#2 classes : healthy not healthy
#1st case

#healthy (10 signals), all with 50 Hz pulse

Healthy_list = np.random.randn(20)
Y_healthy = []
for i in range(0,10):
    Y_healthy.append(Healthy_list[2*i]*np.sin(50 * 2.0*np.pi*x + Healthy_list[2*i + 1]))


#Unhealthy signals, fault = 80 Hz pulse

Unhealthy_list = np.random.randn(20)
Y_unhealthy = []
for i in range(0,10):
    Y_unhealthy.append(Unhealthy_list[2*i]*np.sin(80 * 2.0*np.pi*x + Unhealthy_list[2*i + 1]))


#We calculate the entropy of the basic group
s1_class1 = 10/20
s2_class2 = 10/20

s_entropy = entropy([0.5, 0.5], base = 2)


#2 values for pulse : 50 and 80 Hz

#50 Hz pulse, 

s1_entropy = entropy([1, 0], base = 2)

#80Hz pulse

s2_entropy = entropy([0, 1], base = 2)

gain = s_entropy - 0.5*(s1_entropy + s2_entropy)

#We obtain a gain = 1, which is predictable as all of the info on health
#is contained in the "resonance frequency" feature


#Now the problem is how to determine the values that we take in the algorithm
#For instance if we take the Central frequency, we could have 
#23, 24, 23.5, 23.2, 80, 81, 80.8 Hz and if we don't pay attention it'll
#be treated as 7 values for entropy, when we need to treat it as 2,
#in a more physically coherent way (2 big centers of gravity 23 and 80)

#To do it we'll use frequency binning 
#Even though Freedman_Diaconis method failed because of too few signals -->
#important for DYPE tests


def frequency_binning(amplitudes, frequencies, bin_number):
    
    sorted_indexes = np.argsort(amplitudes)
    
    #This only gives us the indexes of the numbers in the amplitude list
    #when sorted not the sorted list. By doing that we can keep the 
    #frequencies with their respective amplitudes when creating the
    #sorted list right now.
    
    sorted_amplitudes = np.array(amplitudes)[sorted_indexes]
    sorted_frequencies = np.array(frequencies)[sorted_indexes]
    
    #What is hidden below was a tentative of using the Freedman_Diaconis
    #Methode to calculate the number of bins needed. This failed due to 
    #The low number of signals in this bearing dataset example
    #This still is worth looking into for DYPE tests (Cf Python note + report)
    """
    #Now to calculate the number of bins, we'll use the Freedman_Diaconis
    #Method
    
    Q1 = np.percentile(amplitudes, 25)
    Q3 = np.percentile(amplitudes, 75)
    
    if bin_number == 0:
        bin_width = 2*(Q3 - Q1)/(len(amplitudes) ** (1/3))
        bin_number = np.ceil((max(amplitudes) - min(amplitudes)) / bin_width)
    
    """
    bins = np.array_split(sorted_frequencies, bin_number)
    
    freq_bins = np.array_split(sorted_amplitudes, bin_number)
    
    #max_len = max(len(chunk) for chunk in freq_bins)
    
    
    return(freq_bins, bins, bin_number)


def binning(amplitudes, bin):
    amplitudes = np.sort(amplitudes)
    
    amplitudes = np.array_split(amplitudes, bin)
    
    return(amplitudes)
#We'll try an example with the Central_Frequency feature




A = [Mean(y), Variance(y), STD(y), RMS(y), Max(y), Kurtosis(y), 
      Skewness(y), EO(y), Mean_abs(y), SRAV(y), SF(y), IF(y), Crest(y)]

B = [np.mean(yfft), np.var(y), np.std(y), scipy.stats.kurtosis(y), 
     scipy.stats.skew(y), ]



#%%



def information_gain(Calc_0, function, bin_number):
    
    #First we diferentiate the healthy samples from the others
    healthy_feature = {}
    non_healthy_feature = {}
    
    #This little loop is to be sure that for the frequency dictonaries we still
    #Catch the N of the healthy component that's in 5th place "FFT (N...)"
    #compared to the time cases where it's just "N_...."
    a = 5
    for key in Calc_0.keys():
        if key[0] == 'N':
            a = 0
            break
    for key in Calc_0.keys():
    
#        if key == 'N_0' or key == '7_OR':
        if key[a] == 'N':
            healthy_feature[key] = function(Calc_0[key])
        else:
            non_healthy_feature[key] = function(Calc_0[key])
    


#First we calculate the general entropy of the system, based on 
#The repartition healthy/non_healthy
    
    entropy_class = entropy([len(healthy_feature)/(len(healthy_feature) + len(non_healthy_feature)), 
                             len(non_healthy_feature)/(len(healthy_feature) + len(non_healthy_feature))], base = 2)


#Now we only have null entropies if we do 7 values because there
#will always be a class who has no value, if there is only one value and 2 classes

#We need to do bins, for instance 2 or 3, cf report


    #The classes list will contain a one if it's healthy, 0 if 
    #it's unhealthy, to keep track of this for the frequency
    #binning
    classes = []
    amplitudes = []
    for key in healthy_feature.keys():
        classes.append(1)
        amplitudes.append(healthy_feature[key][0])
        #We append the first element because the dictionnary is comprised
        #Of lists of lists so the functiun gives us a one-element list
        #As a result, hence the "[0]"
    
    for key in non_healthy_feature.keys():
        classes.append(0)
        amplitudes.append(non_healthy_feature[key][0])
    
    
    
    A = frequency_binning(amplitudes, classes, bin_number)
    
    #Now that we have bins, we calculate the entropy for each class
    
    entropy_list = []
    for i in range(0, len(A[0])):
        healthy_num = sum(A[1][i])
        fault_num = len(A[1][i]) - healthy_num
        
        healthy_frac = healthy_num/len(A[1][i])
        fault_frac = fault_num/len(A[1][i])
        entropy_list.append(entropy([healthy_frac, fault_frac], base = 2))
        
    split_entropy = 0
    for i in range(0, len(A[0])):
        split_entropy = split_entropy + entropy_list[i]*len(A[1][i])/len(amplitudes)
    
    information_gain = entropy_class - split_entropy
    return(information_gain)


#We create a 2nd function to manage the features of 2 arguments w frequency
#Totally possible to make it into one with a test to check how many variables
#(With a .__code__.co_argcount ==k test inside the function to determine 
#how many arguments we use) and we'll probably have to do this later when
#We tackle features with even more arguments


def information_gain_freq(Calc_FFT, fftFreq, function, bin_number):
    
    #First we diferentiate the healthy samples from the others
    healthy = {}
    non_healthy = {}
    for key in Calc_FFT.keys():
    
#        if key == 'N_0' or key == '7_OR':
        if key[5] == 'N':
            healthy[key] = Calc_FFT[key]
        else:
            non_healthy[key] = Calc_FFT[key]
            
            
    
    
    healthy_feature = {}
    #print(healthy.keys())
    #print(fftFreq.keys())
    #print(Calc_FFT.keys())
    for key in healthy.keys():
        healthy_feature[key] = function(fftFreq[key], healthy[key])
    
    non_healthy_feature = {}
    for key in non_healthy.keys():
        non_healthy_feature[key] = function(fftFreq[key], non_healthy[key])



#First we calculate the general entropy of the system, based on 
#The repartition healthy/non_healthy
    
    entropy_class = entropy([len(healthy)/(len(healthy) + len(non_healthy)), 
                             len(non_healthy)/(len(healthy) + len(non_healthy))], base = 2)


#Now we only have null entropies if we do 7 values because there
#will always be a class who has no value, if there is only one value and 2 classes

#We need to do bins, for instance 2 or 3, cf report

    #The classes list will contain a one if it's healthy, 0 if 
    #it's unhealthy, to keep track of this for the frequency
    #binning
    classes = []
    amplitudes = []
    for key in healthy.keys():
        classes.append(1)
        amplitudes.append(function(fftFreq[key], healthy[key])[0])
        #We append the first element because the dictionnary is comprised
        #Of lists of lists so the functiun gives us a one-element list
        #As a result, hence the "[0]"
    
    for key in non_healthy.keys():
        classes.append(0)
        amplitudes.append(function(fftFreq[key], non_healthy[key])[0])
    
    
    
    A = frequency_binning(amplitudes, classes, bin_number)
    
    #Now that we have bins, we calculate the entropy for each class
    
    entropy_list = []
    for i in range(0, len(A[0])):
        healthy_num = sum(A[1][i])
        fault_num = len(A[1][i]) - healthy_num
        
        healthy_frac = healthy_num/len(A[1][i])
        fault_frac = fault_num/len(A[1][i])
        entropy_list.append(entropy([healthy_frac, fault_frac], base = 2))
        
    split_entropy = 0
    for i in range(0, len(A[0])):
        split_entropy = split_entropy + entropy_list[i]*len(A[1][i])/len(amplitudes)
    
    information_gain = entropy_class - split_entropy
    return(information_gain)







def comparison_info_gain(Calc_0, function1, function2, function3, bin_number):
    A = information_gain(Calc_0, function1, bin_number)
    B = information_gain(Calc_0, function2, bin_number)
    C = information_gain(Calc_0, function3, bin_number)


    return(A,B,C)







def Chi_Square(Calc_0, function, bin_number):
    #First we diferentiate the healthy samples from the others
    healthy = {}
    non_healthy = {}
    
    #This little loop is to be sure that for the frequency dictonaries we still
    #Catch the N of the healthy component that's in 5th place "FFT (N...)"
    #compared to the time cases where it's just "N_...."
    a = 5
    for key in Calc_0.keys():
        if key[0] == 'N':
            a = 0
            break
        
    for key in Calc_0.keys():
    
    #        if key == 'N_0' or key == '7_OR':
        if key[a] == 'N':
            healthy[key] = Calc_0[key]
        else:
            non_healthy[key] = Calc_0[key]
    
    #then we calculate 1 feature, for each sample
    healthy_feature = {}
    for key in healthy.keys():
        healthy_feature[key] = function(healthy[key])
    
    non_healthy_feature = {}
    for key in non_healthy.keys():
        non_healthy_feature[key] = function(non_healthy[key])
    
    
    #The classes list will contain a one if it's healthy, 0 if 
    #it's unhealthy, to keep track of this for the frequency
    #binning
    classes = []
    amplitudes = []
    for key in healthy.keys():
        classes.append(1)
        amplitudes.append(healthy_feature[key][0])
        #We append the first element because the dictionnary is comprised
        #Of lists of lists so the functiun gives us a one-element list
        #As a result, hence the "[0]"
    
    for key in non_healthy.keys():
        classes.append(0)
        amplitudes.append(non_healthy_feature[key][0])
        
        
    A = frequency_binning(amplitudes, classes, bin_number)
     
    #Now that we have bins, let's make the observed frequencies matrix
    #2 columns for 2 classes : healthy + unhealthy
    #Number of lines = Number of bins
    
    observed_matrix = np.zeros((bin_number, 2))
    
    #len(A[1]) = bin_number
    for i in range(0, len(A[1])):
        observed_matrix[i][0] = sum(A[1][i]) #Number of healthy data in this dataset
        observed_matrix[i][1] = len(A[1][i]) - sum(A[1][i])
    
    row_tot = 0
    for i in range(0, bin_number):
        row_tot = row_tot + sum(observed_matrix[i])
        
    
    #Now we create the expected frequencies matrix
    #For the formula used cf doc (Using total rows of lines/colums)
    
    expected_matrix = np.zeros((bin_number, 2))
    
    for i in range(0, bin_number):
            for j in range(0, 2):    
                expected_matrix[i][j] = sum(observed_matrix[i])*sum(observed_matrix[:,j])/row_tot
    
        
    
    #Now we calculate X^2, with the 2 matrixes
    #For the formula cf docu
    
    
    X2 = 0
    for i in range(0, bin_number):
        for j in range(0, 2):
            X2 = X2 + ((observed_matrix[i][j] - expected_matrix[i][j]) ** 2 )/expected_matrix[i][j]
    
    #Now we calculate the degrees of freedom
    
    df = (np.shape(expected_matrix)[0] - 1) * (np.shape(expected_matrix)[1] - 1)
    
    #Now to calculate the p_value
    p_value = chi2.sf(X2, df)
    return([X2, p_value])



def Chi_Square_freq(Calc_FFT, fftFreq, function, bin_number):
    #First we diferentiate the healthy samples from the others
    healthy = {}
    non_healthy = {}
    
    #This little loop is to be sure that for the frequency dictonaries we still
    #Catch the N of the healthy component that's in 5th place "FFT (N...)"
    #compared to the time cases where it's just "N_...."
    a = 5
    for key in Calc_FFT.keys():
        if key[0] == 'N':
            a = 0
            break
        
    for key in Calc_FFT.keys():
    
    #        if key == 'N_0' or key == '7_OR':
        if key[a] == 'N':
            healthy[key] = Calc_FFT[key]
        else:
            non_healthy[key] = Calc_FFT[key]
    
    #then we calculate 1 feature, for each sample
    healthy_feature = {}
    for key in healthy.keys():
        healthy_feature[key] = function(fftFreq[key], healthy[key])
    
    non_healthy_feature = {}
    for key in non_healthy.keys():
        non_healthy_feature[key] = function(fftFreq[key], non_healthy[key])
    
    
    #The classes list will contain a one if it's healthy, 0 if 
    #it's unhealthy, to keep track of this for the frequency
    #binning
    classes = []
    amplitudes = []
    for key in healthy.keys():
        classes.append(1)
        amplitudes.append(healthy_feature[key][0])
        #We append the first element because the dictionnary is comprised
        #Of lists of lists so the functiun gives us a one-element list
        #As a result, hence the "[0]"
    
    for key in non_healthy.keys():
        classes.append(0)
        amplitudes.append(non_healthy_feature[key][0])
        
        
    A = frequency_binning(amplitudes, classes, bin_number)
     
    #Now that we have bins, let's make the observed frequencies matrix
    #2 columns for 2 classes : healthy + unhealthy
    #Number of lines = Number of bins
    
    observed_matrix = np.zeros((bin_number, 2))
    
    #len(A[1]) = bin_number
    for i in range(0, len(A[1])):
        observed_matrix[i][0] = sum(A[1][i]) #Number of healthy data in this dataset
        observed_matrix[i][1] = len(A[1][i]) - sum(A[1][i])
    
    row_tot = 0
    for i in range(0, bin_number):
        row_tot = row_tot + sum(observed_matrix[i])
        
    
    #Now we create the expected frequencies matrix
    #For the formula used cf doc (Using total rows of lines/colums)
    
    expected_matrix = np.zeros((bin_number, 2))
    
    for i in range(0, bin_number):
            for j in range(0, 2):    
                expected_matrix[i][j] = sum(observed_matrix[i])*sum(observed_matrix[:,j])/row_tot
    
        
    
    #Now we calculate X^2, with the 2 matrixes
    #For the formula cf docu
    
    
    X2 = 0
    for i in range(0, bin_number):
        for j in range(0, 2):
            X2 = X2 + ((observed_matrix[i][j] - expected_matrix[i][j]) ** 2 )/expected_matrix[i][j]
    
    #Now we calculate the degrees of freedom
    
    df = (np.shape(expected_matrix)[0] - 1) * (np.shape(expected_matrix)[1] - 1)
    
    #Now to calculate the p_value
    p_value = chi2.sf(X2, df)
    return([X2, p_value])






#%%

def Pearson_Corr(Calc_0, function):
    #First we diferentiate the healthy samples from the others
    healthy = {}
    non_healthy = {}
    a = 5
    for key in Calc_0.keys():
        if key[0] == 'N':
            a = 0
            break
        
    for key in Calc_0.keys():
    
    #        if key == 'N_0' or key == '7_OR':
        if key[a] == 'N':
            healthy[key] = Calc_0[key]
        else:
            non_healthy[key] = Calc_0[key]
    
    #then we calculate 1 feature, for each sample
    healthy_feature = {}
    for key in healthy.keys():
        #print(key)
        healthy_feature[key] = function(healthy[key])
    
    non_healthy_feature = {}
    for key in non_healthy.keys():
        non_healthy_feature[key] = function(non_healthy[key])
    
    
    #The classes list will contain a one if it's healthy, 0 if 
    #it's unhealthy, to keep track of this for the frequency
    #binning
    classes = []
    amplitudes = []
    for key in healthy.keys():
        classes.append(1)
        amplitudes.append(function(healthy[key])[0])
        #We append the first element because the dictionnary is comprised
        #Of lists of lists so the functiun gives us a one-element list
        #As a result, hence the "[0]"
    
    for key in non_healthy.keys():
        classes.append(0)
        amplitudes.append(function(non_healthy[key])[0])
        
    return([pearsonr(amplitudes, classes)[0], pearsonr(amplitudes, classes)[1]])
        
    

    
def Pearson_Corr_freq(Calc_FFT, fftFreq, function):
    #First we diferentiate the healthy samples from the others
    healthy = {}
    non_healthy = {}
    for key in Calc_FFT.keys():
    
    #        if key == 'N_0' or key == '7_OR':
        if key[5] == 'N':
            healthy[key] = Calc_FFT[key]
        else:
            non_healthy[key] = Calc_FFT[key]
    

    #then we calculate 1 feature, for each sample
    healthy_feature = {}
    for key in healthy.keys():
        healthy_feature[key] = function(fftFreq[key], healthy[key])
    
    non_healthy_feature = {}
    for key in non_healthy.keys():
        non_healthy_feature[key] = function(fftFreq[key], non_healthy[key])
    
    
    #The classes list will contain a one if it's healthy, 0 if 
    #it's unhealthy, to keep track of this for the frequency
    #binning
    classes = []
    amplitudes = []
    for key in healthy.keys():
        classes.append(1)
        amplitudes.append(function(fftFreq[key], healthy[key])[0])
        #We append the first element because the dictionnary is comprised
        #Of lists of lists so the functiun gives us a one-element list
        #As a result, hence the "[0]"
    
    for key in non_healthy.keys():
        classes.append(0)
        amplitudes.append(function(fftFreq[key], non_healthy[key])[0])
        
    return([pearsonr(amplitudes, classes)[0], pearsonr(amplitudes, classes)[1]])



    
def Comparison_Pearson(Calc_0, function1, function2, function3):
    A = Pearson_Corr(Calc_0, function1)
    print(1)
    B = Pearson_Corr(Calc_0, function2)
    print(1)
    C = Pearson_Corr(Calc_0, function3)
    print(1)
    
    Data = {'Feature': [function1, function2, function3],
            '|r| value': [A[0], B[0], C[0]],
            'p value': [A[1], B[1], C[1]]}
    
    df = pd.DataFrame(Data)
    
    df2 = df.sort_values(by = '|r| value', ascending = False)
    
    return(df2)
    
    



#%%

from collections import defaultdict

def manhattan_distance(a,b):
    return(np.sum(np.abs(np.subtract(a, b))))

def RelieF(Calc_0, Time_features, function):
    
    
    
    #Intialisation of the weights at 0
    weights = 0
    
    n = len(Calc_0.keys())
    
    classes = []
    feature_value= {}
    
    
    a = 5
    for key in Calc_0.keys():
        if key[0] == 'N':
            a = 0
            break
        
        
    for key in Calc_0.keys():
        #This verification was to isolate the healthy datasets in the bearing example
        if key[a] == 'N':
            
            #We store the feature value and health classification info, 0 for a healthy set
            #1 for an unhealthy set
            feature_value[key] = [function(Calc_0[key]), 0]
        else:
            feature_value[key] = [function(Calc_0[key]), 1]
    
    
    
    
    for i in feature_value.keys():
        
        #Now we check for each feature, the distance between its value and the value
        #of the other features (here with the Manhattan's distance)
        
        
        xi = feature_value[i]
        dist = []
        for j in feature_value.keys():
            if i != j:
                xj = feature_value[j]
                
                dist.append([manhattan_distance(xi[0], xj[0]), xj[1]])
        sorted_dist = sorted(dist, key = lambda item: item[0])
            
        
        
    

        #print(AA)
    #Now we update the weights by substracting the near hit (same class)
    #thus penalizing if the hit is too different, and adding the near miss (diff
    #class) thus rewarding if the miss is spread
        i_class = xi[1]
        
        for x in sorted_dist:
            if x[1] == i_class:
                near_hit = x[0]
                break
        
        for x in sorted_dist:
            if x[1] == 1 - i_class:
                near_miss = x[0]
                break
        
        # near_hit = next((x for x in sorted_dist if x[1] == i_class), None)[0]
        # near_miss = next((x for x in sorted_dist if x[1] == 1 - i_class), None)[0]
        
        weights = weights + (near_miss - near_hit)/n
    
    return(weights)
    
    
    
def RelieF_freq(Calc_FFT, fftFreq, Freq_features, function):
    
    
    
    weights = 0
    
    n = len(Calc_FFT.keys())
    
    classes = []
    feature_value= {}
    for key in Calc_FFT.keys():
        if key[5] == 'N':
            
            #On store l'information avec la valeur de la feature, 0 pour healthy
            #1 pour unhealthy
            feature_value[key] = [function(fftFreq[key], Calc_FFT[key]), 0]
        else:
            feature_value[key] = [function(fftFreq[key], Calc_FFT[key]), 1]
            
    for i in feature_value.keys():
        xi = feature_value[i]
        dist = []
        for j in feature_value.keys():
            if i != j:
                xj = feature_value[j]
                
                dist.append([manhattan_distance(xi[0], xj[0]), xj[1]])
        sorted_dist = sorted(dist, key = lambda item: item[0])
            
        
        
    

        #print(AA)
    #Now we update the weights by substracting the near hit (same class)
    #thus penalizing if the hit is too different, and adding the near miss (diff
    #class) thus rewarding if the miss is spread
        i_class = xi[1]
        
        for x in sorted_dist:
            if x[1] == i_class:
                near_hit = x[0]
                break
        
        for x in sorted_dist:
            if x[1] == 1 - i_class:
                near_miss = x[0]
                break
        
        # near_hit = next((x for x in sorted_dist if x[1] == i_class), None)[0]
        # near_miss = next((x for x in sorted_dist if x[1] == 1 - i_class), None)[0]
        
        weights = weights + (near_miss - near_hit)/n
    
    return(weights)


#%%



"""
#Simple mass spring case without gravity

#parameters
m = 1 #100 grams
k = 100 #100 N/M
w0 = np.sqrt(k/m)

N = 100
T = 1/300
a = 0
phi = 0
alpha = 2 #proportionality coefficient between the 2 springs if there's 2 springs

x = np.linspace(0,N*T, N)

# y1 = a + np.cos(w0*np.sqrt(2*alpha)*x + phi)
y = np.sin(w0 * 2.0*np.pi*x)


yfft_complex = fft(y)
yfft = []
#Module
for i in range(0, len(x)):
    yfft.append(np.sqrt(np.real(yfft_complex[i])**2 + np.imag(yfft_complex[i])**2))

yfreq = np.fft.rfftfreq(len(yfft), T)

plt.figure()
plt.plot(x, y)
plt.title('Time spring mass')
plt.show()

yfft = yfft[:len(yfft)//2 +1]
plt.figure()
plt.plot(yfreq, yfft)
plt.title('FFT spring mass')
plt.show()


y_PSD = np.zeros(len(yfft))
for i in range(0, len(y_PSD)):
    y_PSD[i] = yfft[i] ** 2
    



plt.figure()
plt.plot(yfreq, y_PSD)
plt.title('PSD masse ressort')
plt.axvline(x = 9, ymin = 0, ymax = 1, color = 'red')
plt.show()
"""