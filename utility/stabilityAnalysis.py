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

def plotSingleLinearFit(fig, gridloc, pars, results, data, trainingsInd, testInd, behaviors):
    inner_grid = gridspec.GridSpecFromSubplotSpec(len(behaviors), 3,
                subplot_spec=gridloc, hspace=1, wspace=0.25, width_ratios=[3,1,1])
    for lindex, label in enumerate(behaviors):
        #weights, intercept, alpha, _,_ = resultSet[key][fitmethod][label]
        weights = results[label]['weights']
        intercept = results[label]['intercepts']
        if pars['useRank']:
            x = data['Neurons']['rankActivity']
        else:
            x = data['Neurons']['Activity']
        y = data['Behavior'][label]
       
        # calculate y from model
        yPred = np.dot(weights, x) + intercept
        
        yTrain = np.ones(yPred.shape)*np.nan
        yTrain[trainingsInd] = yPred[trainingsInd]
        
        yTest =  np.ones(yPred.shape)*np.nan
        yTest[testInd] = yPred[testInd]
        
        #if random=='random':
        #    yTest = yPred
        # plot training and test set behavior and prediction
        ax1 = plt.Subplot(fig, inner_grid[lindex, 0])
        
        ax1.plot(data['Neurons']['Time'], yTrain, color=colorBeh[label], label = 'Training', alpha =0.4, lw=2)
        ax1.plot(data['Neurons']['Time'], y, color=colorBeh[label], label = 'Behavior', lw=1)
        ax1.plot(data['Neurons']['Time'], yTest, color=colorPred[label], label = r'$R^2$ {0:.2f}'.format(results[label]['scorepredicted']), lw=1)
        ax1.set_xlim(np.percentile(data['Neurons']['Time'], [0,100]))    
        ax1.set_ylabel(names[label])
        if lindex==len(behaviors)-1:
            ax1.set_xlabel('Time (s)')
        
        ax1.legend(loc=(0.0,0.9), ncol = 2)
        fig.add_subplot(ax1)
        
        # show how predictive each additional neuron is
        ax4 = plt.Subplot(fig, inner_grid[lindex, 2])
        ax4.plot(results[label]['cumulativeScore'], color=colorPred[label],marker='o',  markerfacecolor="none",markersize=5)
        ax4.plot(results[label]['individualScore'], color=colorBeh[label],marker='o', markerfacecolor="none", markersize=5)
          
        ax4.set_ylabel(r'$R^2$ score')
        if lindex==len(behaviors)-1:
            ax4.set_xlabel('Number of neurons')
        fig.add_subplot(ax4)
    # plot weights
        
    ax3 = plt.Subplot(fig, inner_grid[:,1])
    for lindex, label in enumerate(behaviors):
        weights = results[label]['weights']
        
        if lindex == 0:
            indices = np.arange(len(x))
            indices = np.argsort(weights)
        rank = np.arange(0, len(weights))
        ax3.fill_betweenx(rank, np.zeros(len(weights)),weights[indices]/np.max(weights), step='pre', color=colorBeh[label], alpha = 0.5)
    
    ax3.set_ylabel('Neuron weights')
    ax3.spines['left'].set_visible(False)
    ax3.set_yticks([])
    fig.add_subplot(ax3)  

folder = "AML32_moving/BrainScanner20170610_105634_MS/"
folder = "AML32_moving/BrainScanner20170613_134800_MS/"
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
###  show pipeline effects one by one
###
################################################

print "Performing LASSO.",
behaviors = ['AngleVelocity', 'Eigenworm3', 'Eigenworm2']
fig = plt.figure('Lasso 1',(2*6.8, 1.7*len(behaviors)))
outer_grid = gridspec.GridSpec(2, 1, hspace=0.25, wspace=0.25)

results = {'LASSO':{}}
results['LASSO'] = dr.runLasso(data, pars, testIndices, trainingsIndices, plot=0, behaviors = behaviors)
# calculate how much more neurons contribute
tmpDict = dr.scoreModelProgression(data, results, testIndices, trainingsIndices, pars, fitmethod = 'LASSO', behaviors = behaviors)
for tmpKey in tmpDict.keys():
    results['LASSO'][tmpKey].update(tmpDict[tmpKey])
print 'Done with Lasso. Plotting...'
print results
gridloc = outer_grid[0]
plotSingleLinearFit(fig, gridloc , pars, results['LASSO'], data, trainingsIndices, testIndices, behaviors)

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