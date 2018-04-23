# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:21:35 2017
compare the impact of rank transforms on data.
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

    
###############################################    
# 
#    load data into dictionary
#
##############################################  
#folder = "SelectDatasets/BrainScanner20170610_105634_linkcopy/"
#folder = "/home/monika/Dropbox/Work/BehaviorPrediction/PredictionCode/SelectDatasets/{}_linkcopy/"
#dataLog = "/home/monika/Dropbox/Work/BehaviorPrediction/PredictionCode/SelectDatasets/description.txt"
folder = "AML32_moving/{}_MS/"
dataLog = "AML32_moving/AML32_datasets.txt"
##### GFP
#folder = "AML18_moving/{}_MS/"
#dataLog = "AML18_moving/AML18_datasets.txt"
# output is stored here
outLoc = "AML32_moving/Analysis/"

dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder)
keyList = np.sort(dataSets.keys())
nWorms = len(keyList)
# results dictionary 
resultDict = {}
for kindex, key in enumerate(keyList):
    resultDict[key] = {}
# analysis parameters

pars ={'nCompPCA':10, # no of PCA components
        'PCAtimewarp':True, #timewarp so behaviors are equally represented
        'trainingCut': 0.6, # what fraction of data to use for training 
        'trainingType': 'middle', # simple, random or middle.select random or consecutive data for training. Middle is a testset in the middle
        'linReg': 'simple', # ordinary or ransac least squares
        'trainingSample': 1, # take only samples that are at least n apart to have independence. 4sec = gcamp_=->24 apart
        'useRank': 0, # use the rank transformed version od neural data
      }

###############################################    
# 
# run PCA and store results
#
##############################################
#%%

print 'running PCA without rank'
for kindex, key in enumerate(keyList):
    resultDict[key]['PCA'] = dr.runPCANormal(dataSets[key], pars)

# overview of PCA results and weights
mp.plotPCAresults(dataSets, resultDict, keyList, pars, flag='PCA')
# correlate PCA with all sorts of stuff
mp.plotPCAcorrelates(dataSets, resultDict, keyList, pars, flag='PCA')
plt.show()
#  plot 3D trajectory of PCA
#
#mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'phase')
#plt.show()
mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'velocity')

mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'turns')
#plt.show()
# save the most prominent 3 neurons for each PCA axis


pars['useRank'] = 1
print 'running PCA with rank'
for kindex, key in enumerate(keyList):
    resultDict[key]['RankPCA'] = dr.runPCANormal(dataSets[key], pars)

# overview of PCA results and weights
mp.plotPCAresults(dataSets, resultDict, keyList, pars, flag='RankPCA')
mp.plotPCAcorrelates(dataSets, resultDict, keyList, pars, flag='RankPCA')
plt.show()
#  plot 3D trajectory of PCA
mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'phase', flag='RankPCA')
#plt.show()
#mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'velocity', flag='RankPCA')

#mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'turns')
plt.show()


## plot neurons against transformed neurons
fig3 = plt.figure('Transformed neurons',(13.6, nWorms*3.4))
outer_grid2 = mp.gridspec.GridSpec(nWorms, 1,hspace=0.25, wspace=0.25)
for kindex, key in enumerate(keyList):
    
    neuro = dataSets[key]['Neurons']
    whichNeurons = range(0,len(neuro['rankActivity']), 15)
    inner_grid = mp.gridspec.GridSpecFromSubplotSpec(1,len(whichNeurons),
    subplot_spec=outer_grid2[kindex], hspace=0.25, wspace=0.5)
    
    for i, n in enumerate(whichNeurons):
        ax1 = plt.Subplot(fig3, inner_grid[i])
        fig3.add_subplot(ax1)
        ax1.set_title('Neuron ID {}'.format(n))
        ax1.scatter(neuro['Activity'][n],neuro['rankActivity'][n]/1.0/len(neuro['rankActivity']), s=2, alpha=0.2)
        ax1.set_ylabel('rank-transformed')
        ax1.set_xlabel('raw activity')
        ax1.set_xlim([np.percentile(neuro['Activity'], [2.5]), np.percentile(neuro['Activity'], [99.5])])
        ax1.set_ylim([0,1])
outer_grid2.tight_layout(fig3)

# plot neurons against transformed neurons

fig3 = plt.figure('neuron Traces',(nWorms*3.4, 6.8))
outer_grid2 = mp.gridspec.GridSpec(1, nWorms,hspace=0.25, wspace=0.25)
for kindex, key in enumerate(keyList):
    results = resultDict[key]['PCA']
    neuro = dataSets[key]['Neurons']
    whichNeurons = np.argsort(results['neuronWeights'][:,0])[-5:]
    inner_grid = mp.gridspec.GridSpecFromSubplotSpec(len(whichNeurons),1,
    subplot_spec=outer_grid2[kindex], hspace=1, wspace=0.5)
    
    for i, n in enumerate(whichNeurons):
        ax1 = plt.Subplot(fig3, inner_grid[i])
        fig3.add_subplot(ax1)
        ax1.set_title('Neuron ID {}'.format(n))
        ax1.plot(neuro['Activity'][n], color=mp.UCblue[0], zorder=-5, lw=1)
        ax1.plot(neuro['rankActivity'][n]/1.0/len(neuro['rankActivity']), color=mp.UCorange[0], alpha=0.9, lw=1)
    #ax1.set_ylabel('rank-transformed')
    fig3.text(0.0,0.5, 'rank-transformed', rotation='vertical', fontsize=14)
    ax1.set_xlabel('raw activity')
        #ax1.set_xlim([np.percentile(neuro['Activity'], [2.5]), np.percentile(neuro['Activity'], [99.5])])
        #ax1.set_ylim([0,1])
    
outer_grid2.tight_layout(fig3)
plt.show()
        
            
    
