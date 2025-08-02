# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 10:18:20 2025

@author: lfernan4
"""

import numpy as np
from sklearn import *

#Time features

def Mean(list):
        
    return([np.mean(list)])


def Variance(amplitudes):
    m = 0
    a = Mean(amplitudes)
    
    for i in range(0,len(amplitudes)):
        m = m + (amplitudes[i] - a) ** 2
    m = m/len(amplitudes)
    
    
    return(m)



def STD(list):
    return(np.sqrt(Variance(list)))




def RMS(list):
    n = len(list)
    m = 0
    
    for i in range(0,n):
        m = m + list[i] ** 2
    m = np.sqrt(m/n)
    
    return(m)




def Max(list):
    return(max(list))


def Kurtosis(list):
    
    m = 0
    n = len(list)
    Q = Mean(list)
    p = 0
    for i in range(0, n):
        m = m + (list[i] - Q) ** 4
        p = p + (list[i] -Q) ** 2
    
    m = len(list) * m
    p = p ** 2
    #print(m)
    
        
    
    
    #print(p)
    return (m/p)




def Skewness(list):
    n = len(list)
    m = 0
    Q = Mean(list)
    
    for i in range(0, n):
        m = m + (list[i] - Q) ** 3
        

    
    denom = n * STD(list) ** 3
    
    return(m/denom)




def EO(list):
    n = len(list)
    
    m = 0
    denom = 0
    Diff_square = np.zeros(len(list) - 1)
    for i in range(0, n - 1):
        Diff_square[i] = list[i+1]**2 - list[i]**2
        
    B = np.mean(Diff_square)
    for i in range(0, n-1):
        m = m + (list[i+1]**2 - list[i]**2 - B)**4
        denom = denom + (list[i+1]**2 - list[i]**2 - B)**2
    
    m = n**2 * m
    
    # denom = 0
    # for i in range(0, n-1):
    #     denom = denom + (list[i+1]**2 - list[i]**2 - np.mean(Diff_square))**2
        
    denom = denom ** 2
    
    return(m/denom)




def Mean_abs(list):
    n = len(list)
    m = 0
    for i in range(0,n):
        m = m + np.abs(list[i])
        
    return(m/n)




def sq_root_abs(list):
    n = len(list)
    m = 0
    
    for i in range(0, n):
        m = m + np.sqrt(np.abs(list[i]))
        
    m = m/n
    m = m **2
    
    return(m)



def shape_factor(amplitudes):
    return(RMS(amplitudes)/Mean_abs(amplitudes))




def impulse_factor(amplitudes):
    return(Max(amplitudes)/Mean_abs(amplitudes))



def crest_factor(amplitudes):
    return(Max(amplitudes)/RMS(amplitudes))



def clearance_factor(amplitudes):
    num = Max(amplitudes)
    
    denom = 0
    for i in range(0, len(amplitudes)):
        denom = denom + np.sqrt(np.abs(amplitudes[i]))
        
    denom = denom/len(amplitudes)
    denom = denom ** 2
    
    return(num/denom)
    
    
"""

    
def CPT1(amplitudes):
    numerator = 0
    denom = 0

    for i in range(0, len(amplitudes)):
        numerator = numerator + np.log(np.abs(amplitudes[i]) + 1)
        
    A = STD(amplitudes) + 1
    denom = len(amplitudes)* np.log(A)
    
    return(numerator/denom)
    
    
    
def CPT2(amplitudes):
    numerator = 0
    denom = 0

    for i in range(0, len(amplitudes)):
        numerator = numerator + np.exp(amplitudes[i])
        
    denom = len(amplitudes) * np.exp(STD(amplitudes))
    
    
    return(numerator/denom)
    
    
    
    
def CPT3(amplitudes):
    numerator = 0
    
    n = len(amplitudes)
    
    for i in range(0, n):
        numerator = numerator + np.sqrt(np.abs(amplitudes[i]))
        
    denom = n * Variance(amplitudes)
    
    return(numerator/denom)
    
    
"""


#UNDER THIS IS NOT VERIFIED UNLESS SPECIFIED SO NEAR THE TITLE

"""


def Mean_Square_Error(amplitudes): 
    n = len(amplitudes)
    Q = Mean(amplitudes)
    m = 0
    
    for i in range(0, n):
        m = m + (amplitudes[i] - Q) ** 2
    
    return(m/n)


"""


def Log_Log_Ratio(amplitudes):
    n = len(amplitudes)
    S = np.log(STD(amplitudes))
    m = 0
    
    for i in range(0, n):
        m = m + np.log(np.abs(amplitudes[i]) + 1)
    
    return(m/S)
    
    
def std_deviation_index(amplitudes):#verified
    A = STD(amplitudes)
    B = Mean_abs(amplitudes)
    
    return(A/B)



def fifth_moment(amplitudes):
    n = len(amplitudes)
    m = 0
    Q = Mean(amplitudes)
    
    for i in range(0, n):
        m = m + (amplitudes[i] - Q) ** 5
        
    return(m/n)
        



def fifth_moment_normalized(amplitudes):
    A = fifth_moment(amplitudes)
    B = STD(amplitudes) ** 5  
    return(A/B)






def sixth_moment(amplitudes):
    n = len(amplitudes)
    m = 0
    Q = Mean(amplitudes)
    
    for i in range(0, n):
        m = m + (amplitudes[i] - Q) ** 6
        
    return(m/n)
    


def Pulse_Index(amplitudes): 
    A = Max(amplitudes)
    B = Mean(amplitudes)
    
    return(A/B)


def Margin_Index(amplitudes):
    A = Max(amplitudes)
    B = sq_root_abs(amplitudes)
    
    return(A/B)

def mean_deviation_ratio(amplitudes):
    return(Mean(amplitudes)/STD(amplitudes))



def difference_variance(amplitudes):
    n = len(amplitudes)
    m = 0
    
    for i in range(0, n-1):
        m = m + (amplitudes[i+1] - amplitudes[i]) ** 2
        
    return(m/(n-2))





def Min(amplitudes):
    return(min(amplitudes))


def peak_value(amplitudes):
    return((Max(amplitudes) - Min(amplitudes))/2)

def Peak_to_peak(amplitudes):
    return(Max(amplitudes) - Min(amplitudes))


def hist_lower_bound(amplitudes):
    n = len(amplitudes)
    Mini = Min(amplitudes)
    Maxi = Max(amplitudes)
    
    if isinstance(amplitudes[0], (list, tuple, np.ndarray)) == True:
        C = (1/2)*(Maxi[0] - Mini[0])/(n-1)
        Output_Min = Mini[0]
    elif isinstance(amplitudes[0], (int, np.float32, complex)) == True:
        C = (1/2)*(Maxi - Mini)/(n-1)
        Output_Min = Mini
    else:
        return('Error occured in hist_lower_bound')

    return([Output_Min - C])



def hist_upper_bound(amplitudes):
    n = len(amplitudes)
    Mini = Min(amplitudes)
    Maxi = Max(amplitudes)
    
    
    if isinstance(amplitudes[0], (list, tuple, np.ndarray)) == True:
        C = (1/2)*(Maxi[0] - Mini[0])/(n-1)
        Output_Max = Maxi[0]
    elif isinstance(amplitudes[0], (int, np.float32, complex)) == True:
        C = (1/2)*(Maxi - Mini)/(n-1)
        Output_Max = Maxi
    else:
        return('Error occured in hist_higher_bound')
    return([Output_Max + C])




def latitude_factor(amplitudes):
    n = len(amplitudes)
    Q = Max(amplitudes)
    
    m = 0
    for i in range(0, n):
        m = m + np.sqrt(np.abs(amplitudes[i]))
        
    m = (m/n) ** 2
    return(Q/m)



def normalized_std(amplitudes):
    Q = STD(amplitudes)
    P = RMS(amplitudes)
    
    if isinstance(amplitudes[0], (list, tuple, np.ndarray)) == True:
        return([Q[0]/P[0]])
    elif isinstance(amplitudes[0], (int, np.float32, complex)) == True:
        return([Q[0]/P])
    else:
        print("type", type(amplitudes[0]))
        print('An error has occured in NNL')


def waveform_indicator(amplitudes):
    Q = Mean(amplitudes)
    P = RMS(amplitudes)
    if isinstance(amplitudes[0], (list, tuple, np.ndarray)) == True:
        return([P[0]/Q[0]])
    elif isinstance(amplitudes[0], (int, np.float32, complex)) == True:
        return([P/Q[0]])
    else:
        print('An error has occured in Waveform_indicators')





def Wilson_amplitude(amplitudes):
    
    n = len(amplitudes)
    p = 0.02
    m = 0
    for i in range(0, n - 1):
        m = m + np.heaviside(np.abs(amplitudes[i] - amplitudes[i + 1]) - p, 1)

    
    return(m)
    
def Zero_crossing_rate(amplitudes):
    n = len(amplitudes)
    m = 0
    
    for i in range(0, n - 1):
        m = m + np.abs(np.sign(amplitudes[i+1]) - np.sign(amplitudes[i]))
        
    return(m/(n-1))
        

def waveform_length(amplitudes):
    n = len(amplitudes)
    
    m = 0
    for i in range(0, n-1):
        m = m + np.abs(amplitudes[i+1] - amplitudes[i])
        
    return(m)
        



def Energy(amplitudes):
    n = len(amplitudes)
    
    m = 0
    for i in range(0, n):
        m = m + np.abs(amplitudes[i]) ** 2
        
    return(m)


def mean_abs_modif1(amplitudes):
    n = len(amplitudes)
    m = 0
    
    for i in range(0, n//4):
        m = m + 0.5 * np.abs(amplitudes[i])

    for i in range(3*n//4 + 1, n):
        m = m + 0.5 * np.abs(amplitudes[i])
        
        
    for i in range(n//4, 3*n//4 + 1):
        m = m + np.abs(amplitudes[i])
        
        
    return(m/n)
        
   
def mean_abs_modif2(amplitudes):
    n = len(amplitudes)
    m = 0
    
    for i in range(0, n//4):
        m = m + (4*i/n) * np.abs(amplitudes[i])

    for i in range(3*n//4 + 1, n):
        m = m + (4*(i-n)/n) * np.abs(amplitudes[i])
        
        
    for i in range(n//4, 3*n//4 + 1):
        m = m + np.abs(amplitudes[i])          
   
    
    return(m/n)
   
    
   
    
   