# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:12:36 2018
plots for  transiently immobilized animal.
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

key = 'BrainScanner20180327_152059_MS/'
folder ='AML70_chip/{}'.format(key)

print folder
# data parameters
dataPars = {'medianWindow':5, # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow':10, # sgauss window for angle velocity derivative. must be odd
            'rotate':True, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5  # gauss window for red and green channel
            }

# analysis parameters
pars ={'nCompPCA':10, # no of PCA components
        'PCAtimewarp':True, #timewarp so behaviors are equally represented
        'trainingCut': 0.7, # what fraction of data to use for training 
        'trainingType': 'middle', # simple, random or middle.select random or consecutive data for training. Middle is a testset in the middle
        'linReg': 'simple', # ordinary or ransac least squares
        'trainingSample': 1, # take only samples that are at least n apart to have independence. 4sec = gcamp_=->24 apart
        'useRank': 0, # use the rank transformed version of neural data for all analyses
        'useDeconv': 0, # use the rank transformed version of neural data for all analyses
        'useClust': 0, # use the clustered neurons transformed version of neural data for all analyses
      }
     
# data dictionary
dataSets = {}
dataSets[key] = dh.loadData(folder, dataPars, ew=1)     
keyList = dataSets.keys()

# results dictionary 
resultDict = {}
for kindex, key in enumerate(keyList):
    resultDict[key] = {}
    

behaviors = ['AngleVelocity','Eigenworm3']
createIndicesTest = 1#True 
overview = 0#False
pca = 0#False
hierclust = False
linreg = False
lasso = 1
elasticnet = 1#True
positionweights = 1#True
resultsPredictionOverview = 1
###############################################    
# 
# create training and test set indices
# 
##############################################
if createIndicesTest:
    for kindex, key in enumerate(keyList):
        resultDict[key] = {'Training':{}}
        for label in behaviors:
            train, test = dr.createTrainingTestIndices(dataSets[key], pars, label=label)
            resultDict[key]['Training'][label] = {'Train':train  }
            resultDict[key]['Training'][label]['Test']=test
    print "Done generating trainingsets"
###############################################    
# 
# some generic data checking plots
#
##############################################
if overview:
    #mp.plotBehaviorNeuronCorrs(dataSets, keyList, behaviors)
    mp.plotBehaviorOrderedNeurons(dataSets, keyList, behaviors)
    mp.plotVelocityTurns(dataSets, keyList)
    mp.plotDataOverview(dataSets, keyList)
    #mp.plotNeurons3D(dataSets, keyList, threed = False)  
    #mp.plotExampleCenterlines(dataSets, keyList, folder)
    plt.show() 
    
###############################################    
# 
# run PCA and store results
#
##############################################
#%%
if pca:
    print 'running PCA'
    for kindex, key in enumerate(keyList):
        resultDict[key]['PCA'] = dr.runPCANormal(dataSets[key], pars, whichPC=0)
        # overview of data ordered by PCA
        mp.plotDataOverview2(dataSets, keyList, resultDict)
        # overview of PCA results and weights
        mp.plotPCAresults(dataSets, resultDict, keyList, pars)
        plt.show()
        mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'time')
        plt.show()
        # color by before and after
        colorBy = np.zeros(dataSets[key]['Neurons']['Activity'].shape[1])
        colorBy[:int(dataSets[key]['Neurons']['Activity'].shape[1]/2.)] = 1
        mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'Immobilization', colorBy = colorBy)
        plt.show()

        plt.show(block=True)
        resultDict['PCA'] = {}
        resultDict['PCA2'] = {}
        # run PCA on each half
        half1 = np.arange(0,1680)
        half2 = np.arange(1680,dataSets[key]['Neurons']['Activity'].shape[1])
        resultDict[key]['PCAHalf1'] = dr.runPCANormal(dataSets[key], pars, whichPC=0, testset = half1)
        resultDict[key]['PCAHalf2'] = dr.runPCANormal(dataSets[key], pars, whichPC=0, testset = half2)
        mp. plotPCAresults(dataSets, resultDict, keyList, pars, flag = 'PCAHalf1', testset=half1)
        mp. plotPCAresults(dataSets, resultDict, keyList, pars, flag = 'PCAHalf2', testset=half2)
        plt.show()
#%%
###############################################    
# 
# linear regression using LASSO
#
##############################################
if lasso:
    print "Performing LASSO.",
    for kindex, key in enumerate(keyList):
        print key
        splits = resultDict[key]['Training']
        resultDict[key]['LASSO'] = dr.runLasso(dataSets[key], pars, splits, plot=1, behaviors = behaviors)
        # calculate how much more neurons contribute
        tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key],splits, pars, fitmethod = 'LASSO', behaviors = behaviors)
        for tmpKey in tmpDict.keys():
            resultDict[key]['LASSO'][tmpKey].update(tmpDict[tmpKey])
        
        tmpDict = dr.reorganizeLinModel(dataSets[key], resultDict[key], splits, pars, fitmethod = 'LASSO', behaviors = behaviors)
        for tmpKey in tmpDict.keys():
            resultDict[key]['LASSO'][tmpKey]=tmpDict[tmpKey]
    
    mp.plotLinearModelResults(dataSets, resultDict, keyList, pars, fitmethod='LASSO', behaviors = behaviors, random = pars['trainingType'])
    plt.show()
    # overview of LASSO results and weights
    #mp.plotPCAresults(dataSets, resultDict, keyList, pars,  flag = 'LASSO')
    #plt.show()
    #  plot 3D trajectory of SVM
    #mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'etho', flag = 'LASSO')
    plt.show()
    
    
    
#%%
###############################################    
# 
# linear regression using elastic Net
#
##############################################
if elasticnet:
    for kindex, key in enumerate(keyList):
        print 'Running Elastic Net',  key
        splits = resultDict[key]['Training']
        resultDict[key]['ElasticNet'] = dr.runElasticNet(dataSets[key], pars,splits, plot=1, behaviors = behaviors)
        # calculate how much more neurons contribute
        tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key], splits,pars, fitmethod = 'ElasticNet', behaviors = behaviors, )
        for tmpKey in tmpDict.keys():
            resultDict[key]['ElasticNet'][tmpKey].update(tmpDict[tmpKey])
    mp.plotLinearModelResults(dataSets, resultDict, keyList, pars, fitmethod='ElasticNet', behaviors = behaviors,random = pars['trainingType'])
    plt.show()


#%%
###############################################    
# 
# overlay neuron projections with relevant neurons
#
##############################################
if positionweights:
    for kindex, key in enumerate(keyList):
        print 'plotting linear model weights on positions',  key
        
    mp.plotWeightLocations(dataSets, resultDict, keyList, fitmethod='ElasticNet')
    plt.show()
#%%
###############################################    
# 
# plot the number of neurons and scatter plot of predictions fo velocity and turns
#
##############################################
if resultsPredictionOverview:
    fitmethod = 'ElasticNet'
    mp.plotLinearModelScatter(dataSets, resultDict, keyList, pars, fitmethod=fitmethod, behaviors = ['AngleVelocity', 'Eigenworm3'], random = 'none')
    # collect the relevant number of neurons
    
    
    noNeur = []
    for key in keyList:
        noNeur.append([resultDict[key][fitmethod]['AngleVelocity']['noNeurons'], resultDict[key][fitmethod]['Eigenworm3']['noNeurons']])
    noNeur = np.array(noNeur)
    plt.figure()
    plt.bar([1,2], np.mean(noNeur, axis=0),yerr=np.std(noNeur, axis=0) )
    plt.scatter(np.ones(len(noNeur[:,0]))+0.5, noNeur[:,0])
    plt.scatter(np.ones(len(noNeur[:,0]))+1.5, noNeur[:,1])
    plt.xticks([1,2], ['velocity', 'Turns'])
    plt.show()
    