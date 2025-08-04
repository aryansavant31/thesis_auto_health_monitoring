# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 10:20:33 2025

@author: lfernan4
"""
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import find_peaks
from scipy.special import erfinv

#Frequency features


def meanF(list):
    return([np.mean(list)])


def varianceF(list):
    n = len(list)
    m = 0
    Q = meanF(list)
    
    
    for i in range(0,n):
        m = m + (list[i] - Q)**2

    return(m/n)
    


def skewnessF(list):
    n = len(list)
    m = 0
    Q = meanF(list)
    
    
    for i in range(0,n):
        m = m + (list[i] - Q)**3
        
    A = np.sqrt(varianceF(list))**3
    denom = n*A
    return(m/denom)





def kurtosisF(list):
    n = len(list)
    m = 0
    denom = 0
    Q = meanF(list)
    
    
    for i in range(0,n):
        m = m + (list[i] - Q)**4
        

    denom = n*varianceF(list)**2
    return(m/denom - 3)







def central_freq(freq, amplitudes):
    n = len(amplitudes)
    m = 0
    denom = 0
    
    
    for i in range(0,n):
        m = m + freq[i]*amplitudes[i]
        denom = denom + amplitudes[i]
        
#    print(m)
    # denom = 0
    # for i in range(0,n):
    #     denom = denom + amplitudes[i]
    
#   print(denom)
    return([m/denom])




def stdF(freq, amplitudes):
    numerator = 0
    denom = 0
    A = central_freq(freq, amplitudes)
    
    n = len(amplitudes)
    
    for i in range(0,n):
        numerator = numerator + (freq[i] - A)**2 * amplitudes[i]
        denom = denom + amplitudes[i]
        
    return(np.sqrt(numerator/denom))
    
    
def rmsF(freq, amplitudes):
    numerator = 0
    denom = 0
    
    for i in range(0, len(amplitudes)):
        numerator = numerator + freq[i]**2 * amplitudes[i]
        denom = denom + amplitudes[i]
    #print(denom)
    return([np.sqrt(numerator/denom)])


"""

def CP1(freq, amplitudes):
    numerator = 0
    denom = 0

    for i in range(0, len(amplitudes)):
        numerator = numerator + (freq[i] - CentralFreq(freq, amplitudes)) ** 3 * amplitudes[i]
        
    denom = len(amplitudes) * STDF(freq, amplitudes) ** 3
    
    return(numerator/denom)
        
        
def CP2(freq, amplitudes):
    return(STDF(freq, amplitudes)/CentralFreq(freq, amplitudes))  


def CP3(freq, amplitudes):
    numerator = 0

    for i in range(0, len(amplitudes)):
        numerator = numerator + (freq[i] - CentralFreq(freq, amplitudes)) ** (1/2) * amplitudes[i]
        
    denom = len(amplitudes) * np.sqrt(STDF(freq, amplitudes))
    
    return(numerator/denom)
        
        
def CP4(freq, amplitudes):
    numerator = 0
    denom = 0

    for i in range(0, len(amplitudes)):
        numerator = numerator + (freq[i] - CentralFreq(freq, amplitudes)) ** 3 * amplitudes[i]
        
    denom = len(amplitudes) * STDF(freq, amplitudes) ** 2
    
    return(numerator/denom)          
        
        
        
def CP5(freq, amplitudes):

    numerator = 0
    denom = 0
    n = len(amplitudes)

    for i in range(0, n):
        numerator = numerator + freq[i]**4 * amplitudes[i]
        denom = denom + freq[i]**2 * amplitudes[i]    
    
    
    return(np.sqrt(CP5))

"""



    

    
    
    
def spectral_spread(freq, amplitudes):
    S = central_freq(freq, amplitudes)
    n = len(amplitudes)
    
    m = 0 
    denom = 0
    
    for i in range(0, n):
        m = m + ((freq[i] - S) ** 2) * amplitudes[i]
        denom = denom + amplitudes[i]
    
    
    if isinstance(m, (list, tuple, np.ndarray)) == True:
        m = m[0]
    if isinstance(denom, (list, tuple, np.ndarray)) == True:
        denom = denom[0]
        
    
    return([np.sqrt(m/denom)])
    
    
def spectral_entropy(amplitudes):
    Q = sum(amplitudes)
    m = 0
    n = len(amplitudes)
    
    for i in range(0, n):
        if amplitudes[i] != 0:
            m = m + amplitudes[i]/Q * np.log2(amplitudes[i]/Q)
    
    return([-1*m])
    





def Kp_value(power):
    #first we find the treshold
    
    alpha = 0.01
    #alpha is the level of error 
    T = 2 * (erfinv(1 - alpha)) ** 2

    
    Kp = find_peaks(power, height = T)
    return(Kp)


def total_power(amplitudes):
    
    power = 2*(amplitudes ** 2)
    Kp = Kp_value(power)

    m = 0
    for i in Kp[0]:
        m = m + power[i]
    return([m])
    

def median_freq(freq, amplitudes):
    power = 2*(amplitudes ** 2)
    Kp = Kp_value(power)
    Q = total_power(amplitudes)[0]
    
    m = 0
    flag = 0
    for i in Kp[0]:
        m = m + power[i]
        if m >= Q/2:
            flag = i
            break

    return([freq[flag]])
        

    
    
def pkF(amplitudes):
    return([max(amplitudes ** 2)])
    












#MAYBE NORMALIZE BY THE TOTAL POWER TO GET THE FREQUENCY AROUND WICH THE ENERGY
#IS CONCENTRATED




def first_spectral_moment(freq, amplitudes):
    power = 2*(amplitudes ** 2)
    Kp = Kp_value(power)
    m = 0
    Q = total_power(amplitudes)[0]
    #print(Kp)
    
    for i in Kp[0]:
        m = m + freq[i]*power[i]
    
    if isinstance(m, (list, tuple, np.ndarray)) == True:
        m = m[0]
        
    #Because of the segmentation of the samples sometimes the first one has no 
    #peaks and so a Q value of 0
    if Q ==0:
        return([0])
    return([m/Q])


def second_spectral_moment(freq, amplitudes):
    power = 2*(amplitudes ** 2)
    Kp = Kp_value(power)
    m = 0
    Q = total_power(amplitudes)[0]
    
    for i in Kp[0]:
        m = m + (freq[i] ** 2) * power[i]

    if isinstance(m, (list, tuple, np.ndarray)) == True:
        m = m[0]
        
    if Q ==0:
        return([0])
    return([m/Q])




def third_spectral_moment(freq, amplitudes):
    power = 2*(amplitudes ** 2)
    Kp = Kp_value(power)
    m = 0
    Q = total_power(amplitudes)[0]
    
    for i in Kp[0]:
        m = m + (freq[i] ** 3) * power[i]

    if isinstance(m, (list, tuple, np.ndarray)) == True:
        m = m[0]
        
    if Q ==0:
        return([0])
    return([m/Q])



def fourth_spectral_moment(freq, amplitudes):
    power = 2*(amplitudes ** 2)
    Kp = Kp_value(power)
    m = 0
    Q = total_power(amplitudes)[0]
    
    
    for i in Kp[0]:
        m = m + (freq[i] ** 4) * power[i]

    if isinstance(m, (list, tuple, np.ndarray)) == True:
        m = m[0]
        
    if Q ==0:
        return([0])
    return([m/Q])




def vcf(freq, amplitudes):
    Q = total_power(amplitudes)[0]
    if Q ==0:
        return([0])
    
    else:
        A1 = second_spectral_moment(freq, amplitudes)[0]/Q
        A2 = first_spectral_moment(freq, amplitudes)[0]/Q
    
        A2 = A2 ** 2
    
    return([A1 - A2])



def frequency_ratio(amplitudes):
    n = len(amplitudes)
    numerator = 0
    denom = 0
    power = 2*(amplitudes ** 2)
    
    for i in range(0, n//2):
        numerator = numerator + power[i]
    
    for i in range(n//2 + 1, n):
        denom = denom + power[i]
        
    #print([n, denom, numerator])
    
    #Here we take the assumption that if the denominator is 0, that 
    #Means the signal is not significant or too weak to be exploitable
    #And we return a ratio of 0 even though it should be infinite
    #9usually if the denominator is null the numerator is also 
    #almost null making the ratio calculation less coherent
    
    if denom == 0:
        return([0])
    else:
        return([numerator/denom])
    
    
def hsc(freq, amplitudes):
    
    power = 2*(amplitudes ** 2)
    Kp = Kp_value(power)
    
    m = 0
    denom = 0
    
    
    for i in Kp[0]:
        m = m + freq[i]*amplitudes[i]
        denom = denom + amplitudes[i]
        
    if denom == 0:
        return([0])
    return([m/denom])


def spectral_flux(amplitudes):
    power = 2*(amplitudes ** 2)
    
    m = 0
    n = len(amplitudes)
    
    for i in range (0, n-1):
        m = m + np.abs((power[i] - power[i+1])**2)

    return([np.sqrt(m)])


def rolloff_frequency_90(freq, amplitudes):
    
    power = 2*(amplitudes ** 2)
    
    
    Q = total_power(amplitudes)[0]*0.9
    m = 0
    n = len(amplitudes)
    for i in range(0, n):
        m = m + power[i]
        if m >Q:
            flag = i
            break
        
    return([freq[flag]])



def rolloff_frequency_85(freq, amplitudes):
    
    power = 2*(amplitudes ** 2)
    
    
    Q = total_power(amplitudes)[0]*0.85
    m = 0
    n = len(amplitudes)
    for i in range(0, n):
        m = m + power[i]
        if m >Q:
            flag = i
            break
        
    return([freq[flag]])
        


def rolloff_frequency_75(freq, amplitudes):
    
    power = 2*(amplitudes ** 2)
    
    
    Q = total_power(amplitudes)[0]*0.75
    m = 0
    n = len(amplitudes)
    for i in range(0, n):
        m = m + power[i]
        if m >Q:
            flag = i
            break
        
    return([freq[flag]])


def rolloff_frequency_95(freq, amplitudes):
    
    power = 2*(amplitudes ** 2)
    
    
    Q = total_power(amplitudes)[0]*0.95
    m = 0
    n = len(amplitudes)
    for i in range(0, n):
        m = m + power[i]
        if m >Q:
            flag = i
            break
        
    return([freq[flag]])

def upper_limit_harmonicity(freq, amplitudes):
    
    power = 2*(amplitudes ** 2)
    Kp = Kp_value(power)
    
    n = len(Kp[0])
    if n == 0:
        return([0])
    return([freq[Kp[0][n-1]]])
