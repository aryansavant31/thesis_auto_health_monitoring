# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:17:53 2025

@author: lfernan4
"""

import scipy.io
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft

#Path of the file of the Python_files

#print(os.getcwd())
path0 = os.path.join(os.getcwd(), 'datasets', 'raw_data', 'bearing_cwru_raw_data')
#print(path0)

#We create a dictionnary having for all faults, only the numerical data we want
#And to be able to manipulate those datas, calculate features

#Dictionnary where there are all original signals for the calculations
Calc_0 = {}

for root, dirs, files in os.walk(path0, topdown = False):
    for file_name in files:
        path = os.path.join(root, file_name)
        #print(path)
        
        
        mat = scipy.io.loadmat(path)
        
        #We get for each fault file, the signal that interests us
        key_name = list(mat.keys())[3]
        DE_data = mat.get(key_name)
        
        #But we stock it in the dictionary under the fault label, not the 
        #label it had in the fault file
        
        Calc_0[file_name[:-4]] = DE_data
        
spacing = 1/48000
#test

x = np.linspace(0, len(DE_data)*spacing, len(DE_data))
y = np.sin(x*np.pi*2)

yfft = fft(y)
yfreq = np.fft.rfftfreq(len(yfft), spacing)

#Calc_0['sinus'] = y

        
#Calculate the Fourier Transforms and keep only the module

Calc_FFT = {}
Calc_FFT_complex = {}
for key in Calc_0.keys():
    
    #We create an intermediate complex dictionary to put the FFT
    #To have a final FFT dictionary with real values corresponding to the 
    #Module of the fft, and that can be plotted (imaginary values
    #work really poorly with plt.plot)
    Calc_FFT_complex['FFT (' + key + ')'] = fft(Calc_0[key])
    Calc_FFT['FFT (' + key + ')'] = np.zeros(len(Calc_FFT_complex['FFT (' + key + ')']))
    for i in range(0, len(Calc_FFT_complex['FFT (' + key + ')'])):
        # if i == len(Calc_FFT['FFT (' + key + ')'])-1:
        #     print( Calc_FFT['FFT (' + key + ')'][i])
        Calc_FFT['FFT (' + key + ')'][i] = np.sqrt( np.real(Calc_FFT_complex['FFT (' + key + ')'][i])**2 + np.imag(Calc_FFT_complex['FFT (' + key + ')'][i])**2)
        # if i == len(Calc_FFT['FFT (' + key + ')'])-1:
        #     print( Calc_FFT['FFT (' + key + ')'][i])
    
#Calculate the PSD

Calc_PSD = {}
for key in Calc_0.keys():
    Calc_PSD['PSD' + key] = np.abs(Calc_FFT['FFT (' + key + ')']) ** 2



#Get frequencies that correspond to FFT
#Can calc with either PSD or FFT

fftFreq = {}

for key in Calc_FFT.keys():
    fftFreq[key] = np.fft.rfftfreq(len(Calc_FFT[key]), spacing)
    
    
    
#Adjust the PSD and FFT calculations to get rid of the negative part
#And to match the frequencies
    
for key in Calc_PSD.keys():
    Calc_PSD[key] = Calc_PSD[key][:len(Calc_PSD[key])//2 +1]

for key in Calc_FFT.keys():
    Calc_FFT[key] = Calc_FFT[key][:len(Calc_FFT[key])//2 +1]

    
win_len = 1000
stride = 200

X = []
Y = []

df = pd.read_csv('all_faults.csv')


for k in df['fault'].unique():
    
    df_temp_2 = df[df['fault'] == k]
    
    for i in np.arange(0, len(df_temp_2) - win_len, stride):
        temp = df_temp_2.iloc[i:i+win_len,:1].values
        #This gives us our window but as a column, a (n,1)matrix
        temp = temp.reshape((1,-1))
        #This reshapes it as a line, a (1, n) matrix
        #The minus 1 is for Python to automatically calculate 
        #The number n
        X.append(temp)
        Y.append(df_temp_2.iloc[i+win_len,-1])
        

X = np.array(X)

X = X.reshape((X.shape[0], win_len))
Y = np.array(Y)

if __name__ == '__main__':
    
    plt.figure()
    plt.plot(X[0])
    plt.figure()
    plt.plot(X[1])
    plt.show()
    
    for f in df['fault'].unique():
        plt.figure(figsize=(10,3))
        plt.plot(df[df['fault']==f].iloc[:,0])
        plt.title(f)
        plt.show()





if __name__ == '__main__':    
    #Create the panda Dataframe with all the curves   
    
        
    df = pd.DataFrame(columns=['DE_data', 'fault'])
        
    for key in Calc_FFT.keys():
        fault = np.full((len(Calc_FFT[key]), 1), key)
        df_temp = pd.DataFrame({'DE_data':np.ravel(Calc_FFT[key]), 'fault':np.ravel(fault)})
        
        df = pd.concat([df, df_temp], axis = 0)
        
    
    
    df.to_csv('all_faults.csv', index=False)
    
    
    
    df = pd.read_csv('all_faults.csv')
    faults = df['fault'].unique()
    
    
    spacing = 1/48000
    
    
    
    
    
    for f in faults:
        plt.figure(figsize=(10,3))
        plt.plot(fftFreq[f], df[df['fault']==f].iloc[:,0])
        plt.title(f)
        plt.show()
    
    # plt.figure(figsize=(10,3))
    # plt.plot(fftFreq['FFT (sinus)'][:10], df[df['fault']=='FFT (sinus)'].iloc[:10,0])
    # plt.title(f)
    # plt.show()
    
    #test
    
    plt.figure()
    plt.plot(yfreq[0:10], yfft[0:10])
    plt.show()

        