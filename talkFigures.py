# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:38:52 2018
make LASSO/EN figures.
@author: monika
"""


# standard modules
import numpy as np
import matplotlib.pylab as plt
import h5py
# custom modules 
import dataHandler as dh
import makePlots as mp
import dimReduction as dr
# scikit learn
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

def rolling_window(a, window):
    a = np.pad(a, (0,window), mode="constant", constant_values=(np.nan,))
    shape = a.shape[:-1] + (a.shape[-1] - window, window)
    strides = a.strides + (a.strides[-1],)
    
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def chunk_window(a, window):
    rest = len(a)%window
    if rest !=0:
        a = np.pad(a, (0,window-rest), mode="constant")
    return a.reshape((-1, window))

def equalTrainsets(data, testsize):
    """given a dataset, create an equlized set with test/train split"""
    
    # calculate histogram of overall data
    nbin = 10
    hist, bins = np.histogram(data, nbin)
    # create blocks of data and calculate entropy => more is more interesting
    chunkyData = rolling_window(data, window=testsize)
    # calculate the entropy of each block
    ent = 
    center = np.where(np.abs(data['Behavior']['Eigenworm3'])>15)[0]
    print center
    testsize = int((1-pars['trainingCut'])*timeLen/2.)
    testLoc = np.random.randint(0,len(center))
    testIndices = np.arange(center[testLoc]-testsize, np.min([len(data['Behavior']['Eigenworm3']),center[testLoc]+testsize]))
#        cutoff = int(pars['trainingCut']*timeLen)
#        testIndices = np.arange(timeLen)[cutoff:]
    #cutoff = int((pars['trainingCut'])*timeLen/2.)
    #tmpIndices = np.arange(timeLen)
    #testIndices = tmpIndices[cutoff:-cutoff]
    # create a trainingset by equalizing probabilities
    # bin  to get probability distribution
    nbin = 10
    hist, bins = np.histogram(data['Behavior'][label], nbin)
    # this is the amount of data that will be left in each bin after equalization
    N = np.sum(hist)/50.#hist[0]+hist[-1]
    # digitize data 
    dataProb = np.digitize(data['Behavior'][label], bins=bins[:-2], right=True)
    # rescale such that we get desired trainingset length
    trainingsIndices= []
    
    tmpTime = np.arange(0,timeLen)
    
    np.random.shuffle(tmpTime)
    counter = np.zeros(hist.shape)
    for index in tmpTime:
            if index not in testIndices:
                # enrich for rare stuff
                n = dataProb[index]
                if counter[n] <= N:
                    trainingsIndices.append(index)
                    counter[n] +=1
    print len(trainingsIndices)/1.0/timeLen, len(testIndices)/1.0/timeLen
    plt.hist(data['Behavior'][label], normed=True,bins=nbin )
    plt.hist(data['Behavior'][label][trainingsIndices], normed=True, alpha=0.5, bins=nbin)
    plt.show()
    return np.sort(trainingsIndices), np.sort(testIndices)    
    
###############################################    
# 
#    load data into dictionary
#
##############################################  
# data parameters
dataPars = {'medianWindow':13, # smooth eigenworms with median filter of that size, must be odd
            'savGolayWindow':13, # savitzky-golay window for angle velocity derivative. must be odd
            'rotate':True # rotate Eigenworms using previously calculated rotation matrix
            }
behaviors = ['AngleVelocity', 'Eigenworm3', 'Eigenworm2']

#folder = Performing LASSO. BrainScanner20170424_105620
folder = "AML32_moving/BrainScanner20170610_105634_MS/"
data = dh.loadData(folder, dataPars)
###############################################    
# 
#    create good test/trainingsets for each behavior
#
############################################## 
test_train_split = {}
for beh in behaviors:
    