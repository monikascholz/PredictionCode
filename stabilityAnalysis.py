# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:46:39 2017
Test the stability and optimal parameters for LASSO and elastic net. Also test 
@author: monika
"""

# standard modules
import numpy as np
import matplotlib.pylab as plt
import h5py
from sklearn.metrics import r2_score
import matplotlib.gridspec as gridspec
# custom modules 
import dataHandler as dh
import makePlots as mp
import dimReduction as dr

folder = "AML32_moving/BrainScanner20170610_105634_MS/"
dataLog = "AML32_moving/AML32_datasets.txt"
##### GFP
#folder = "AML18_moving/{}_MS/"
#dataLog = "AML18_moving/AML18_datasets.txt"
# output is stored here
outLoc = "AML32_moving/Analysis/"

# data parameters
dataPars = {'medianWindow':13, # smooth eigenworms with median filter of that size, must be odd
            'savGolayWindow':13, # savitzky-golay window for angle velocity derivative. must be odd
            'rotate':True,  # rotate Eigenworms using previously calculated rotation matrix
            'savGolayWindowGCamp': 13 # savitzky-golay window for red anc green channel
            }
data = dh.loadData(folder, dataPars)
# keep test and training fixed
tmpIndices = np.arange(len(data['Neurons']['Time']))
# eyeballing some diverse regions, second block includes turns

#testIndices = np.concatenate([np.arange(0,1500 ), np.arange(2000, 3300)])
#trainingsIndices = np.setdiff1d(tmpIndices, testIndices)
#testIndices = np.arange(2000, 3300)\
#trainingsIndices = np.arange(1000,1800)
#testIndices = np.arange(1800,2403)

testIndices = np.concatenate([np.arange(0,500 ), np.arange(1000,1900)])
trainingsIndices = np.setdiff1d(tmpIndices, testIndices)

pars ={'nCompPCA':10, # no of PCA components
        'PCAtimewarp':True, #timewarp so behaviors are equally represented
        'trainingCut': 0.4, # what fraction of data to use for training 
        'trainingType': 'middle', # simple, random or middle.select random or consecutive data for training. Middle is a testset in the middle
        'linReg': 'simple', # ordinary or ransac least squares
        'trainingSample': 1, # take only samples that are at least n apart to have independence. 4sec = gcamp_=->24 apart
        'useRank': 0, # use the rank transformed version of neural data for all analyses
      }


#################################################    
### 
###  show pipeline effects
###
################################################

print "Performing LASSO.",
behaviors = ['AngleVelocity', 'Eigenworm3', 'Eigenworm2']
fig = plt.figure('Lasso 1',(2*6.8, 1.7*len(behaviors)))
outer_grid = gridspec.GridSpec(1, 1, hspace=0.25, wspace=0.25)

results = {'LASSO':{}}
results['LASSO'] = dr.runLasso(data, pars, testIndices, trainingsIndices, plot=0, behaviors = behaviors)
# calculate how much more neurons contribute
tmpDict = dr.scoreModelProgression(data, results, testIndices, trainingsIndices, pars, fitmethod = 'LASSO', behaviors = behaviors)
for tmpKey in tmpDict.keys():
    results['LASSO'][tmpKey].update(tmpDict[tmpKey])
print 'Done with Lasso. Plotting...'
print results
mp.plotSingleLinearFit(fig, outer_grid[0], pars, results['LASSO'], data, trainingsIndices, testIndices, behaviors)

plt.figure('Lasso 1')



plt.show()
   

#es
#################################################    
### 
###   plot velocity versus cms velocity
###
################################################
#medians = np.arange(7,25,2)
#medianFiltData = np.zeros((len(medians),3))
#beh = 'AngleVelocity'
#for ind, medfilt in enumerate(medians):
#    dataPars['savGolayWindow'] = medfilt
#    data = dh.loadData(folder, dataPars)
#    cms = data['Behavior']['CMSVelocity']
#    
#    av = data['Behavior']['AngleVelocity']# in mm/s ?
#    cms = cms/np.max(cms)*np.max(av)
#    print 'savgo', medfilt, r2_score(cms, av) 
#    plt.subplot(211)
#    plt.scatter(cms, av, alpha=0.1)
#    plt.ylabel('Angle Velocity')
#    plt.xlabel('CMS Velocity')
#    plt.subplot(212)
#    plt.plot(cms, label='CMS velocity')
#    plt.plot(av,  label='Angle Velocity', alpha=0.7)
#    plt.legend()
#    plt.show()      
     
### play with LASSO

beh = 'Eigenworm3'

data = dh.loadData(folder, dataPars)
results = dr.runLasso(data, pars, testIndices, trainingsIndices, plot = 1, behaviors = [beh])#, 'Eigenworm3'])
results = dr.runElasticNet(data, pars, testIndices, trainingsIndices, plot = 1, behaviors = [beh])#, 'Eigenworm3'])
 
     
###############################################    
# 
#   Test how stable LASSO is in response to median filtering
#
##############################################
medians = np.arange(1,15,2)
medianFiltData = np.zeros((len(medians),3))
beh = 'Eigenworm3'
dataPars['savGolayWindow'] = 13
for ind, medfilt in enumerate(medians):
    dataPars['medianWindow'] = medfilt
    data = dh.loadData(folder, dataPars)
    results = dr.runLasso(data, pars, testIndices, trainingsIndices, plot = 1, behaviors = [beh])#, 'Eigenworm3'])
    results = dr.runElasticNet(data, pars, testIndices, trainingsIndices, plot = 1, behaviors = [beh])#, 'Eigenworm3'])
    medianFiltData[ind]= medfilt, results[beh]['scorepredicted'], results[beh]['noNeurons']
    
plt.subplot(211)
plt.plot(medianFiltData[:,0], medianFiltData[:,1])
plt.ylabel('R2 score')
plt.subplot(212)
plt.plot(medianFiltData[:,0], medianFiltData[:,2])
plt.ylabel('Number of neurons')
plt.xlabel('median filter size')
plt.show()

###############################################    
# 
#   Test how stable LASSO is in response to sav golay
#
##############################################
medians = np.arange(7,25,2)
medianFiltData = np.zeros((len(medians),3))
dataPars['medianWindow'] = 13
beh = 'AngleVelocity'
for ind, medfilt in enumerate(medians):
    dataPars['savGolayWindow'] = medfilt
    data = dh.loadData(folder, dataPars)
    results = dr.runLasso(data, pars, testIndices, trainingsIndices, plot = 0, behaviors = [beh])#, 'Eigenworm3'])
    medianFiltData[ind]= medfilt, results[beh]['scorepredicted'], results[beh]['noNeurons']

plt.subplot(211)
plt.plot(medianFiltData[:,0], medianFiltData[:,1])
plt.ylabel('R2 score')
plt.subplot(212)
plt.plot(medianFiltData[:,0], medianFiltData[:,2])
plt.ylabel('Number of neurons')
plt.xlabel('Savitzky-Golay filter size')
plt.show()