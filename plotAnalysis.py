
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
outLoc = "AML32_moving/Analysis/Results.hdf5"
##### GFP
folderCtrl = "AML18_moving/{}_MS/"
dataLogCtrl = "AML18_moving/AML18_datasets.txt"
outLoc2 = "AML18_moving/Analysis/Results.hdf5"
# data parameters
dataPars = {'medianWindow':3, # smooth eigenworms with gauss filter of that size, must be odd
            'savGolayWindow':5, # savitzky-golay window for angle velocity derivative. must be odd
            'rotate':True, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5 # gauss window for red and green channel
            }


dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars)
keyList = np.sort(dataSets.keys())
resultDict = dh.loadDictFromHDF(outLoc) 
resultDictCtrl = dh.loadDictFromHDF(outLoc2)
dataSetsCtrl = dh.loadMultipleDatasets(dataLogCtrl, pathTemplate=folderCtrl, dataPars = dataPars)
keyListCtrl = np.sort(dataSetsCtrl.keys())
# analysis parameters

pars ={'nCompPCA':10, # no of PCA components
        'PCAtimewarp':True, #timewarp so behaviors are equally represented
        'trainingCut': 0.6, # what fraction of data to use for training 
        'trainingType': 'middle', # simple, random or middle.select random or consecutive data for training. Middle is a testset in the middle
        'linReg': 'simple', # ordinary or ransac least squares
        'trainingSample': 1, # take only samples that are at least n apart to have independence. 4sec = gcamp_=->24 apart
        'useRank': 0, # use the rank transformed version of neural data for all analyses
      }

behaviors = ['AngleVelocity', 'Eigenworm3']#, 'Eigenworm2']
#behaviors = ['AngleVelocity']

###############################################    
# 
# check which calculations to perform
#
##############################################
createIndicesTest = True 
overview = 0#False
svm = 0
pca = 0#False
hierclust = False
linreg = False
lasso = 1
elasticnet = 0#True
positionweights = 1#True
resultsPredictionOverview = 1

###############################################    
# 
# some generic data checking plots
#
##############################################
if overview:
    #mp.plotBehaviorAverages(dataSets, keyList) 
    mp.plotVelocityTurns(dataSets, keyList)
    mp.plotDataOverview(dataSets, keyList)
    mp.plotNeurons3D(dataSets, keyList, threed = False)  
    #mp.plotExampleCenterlines(dataSets, keyList, folder)
    plt.show() 
    
###############################################    
# 
# average results
#
##############################################
mp.averageResultsLinear(resultDict,resultDictCtrl, keyList,keyListCtrl, fitmethod = "LASSO",  behaviors = ['AngleVelocity', 'Eigenworm3'])
plt.show()
#mp.averageResultsLinear(resultDict, resultDictCtrl, keyList,keyListCtrl, fitmethod = "ElasticNet",  behaviors = ['AngleVelocity', 'Eigenworm3'])
#plt.show()
mp.averageResultsPCA(resultDict, resultDictCtrl, keyList,keyListCtrl, fitmethod = "PCA")
plt.show()
###############################################    
# 
# use svm to predict discrete behaviors
#
##############################################
if svm:
    
    # overview of SVM results and weights
    mp.plotPCAresults(dataSets, resultDict, keyList, pars,  flag = 'SVM')
    plt.show()
    #  plot 3D trajectory of SVM
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'etho', flag = 'SVM')
    plt.show()
#        mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'time',  flag = 'SVM')
#        plt.show()
#        mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'velocity',  flag = 'SVM')
#        plt.show()
#        mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'turns',  flag = 'SVM')
#        plt.show()

###############################################    
# 
# run PCA and store results
#
##############################################
#%%
if pca:
    
    # overview of PCA results and weights
    mp.plotPCAresults(dataSets, resultDict, keyList, pars)
    plt.show()
    # show correlates of PCA
    mp.plotPCAcorrelates(dataSets, resultDict, keyList, pars, flag='PCA')
    #  plot 3D trajectory of PCA
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'etho')
    plt.show()
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'time')
    plt.show()
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'velocity')
    plt.show()
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'turns')
    plt.show()
    #%%
    ###############################################    
    # 
    # overlay neuron projections with relevant neurons
    #
    ##############################################
    # TODO change to make weights larger or stth.
    
        
    #mp.plotWeightLocations(dataSets, resultDict, keyList, fitmethod='PCA')
    #plt.show()
    #%%
  
#%%
###############################################    
# 
# linear regression using LASSO
#
##############################################
if lasso:
    
    mp.plotLinearModelResults(dataSets, resultDict, keyList, pars, fitmethod='LASSO', behaviors = behaviors, random = pars['trainingType'])
    plt.show()
    # overview of LASSO results and weights
    mp.plotPCAresults(dataSets, resultDict, keyList, pars,  flag = 'LASSO')
    plt.show()
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
    
    mp.plotLinearModelResults(dataSets, resultDict, keyList, pars, fitmethod='ElasticNet', behaviors = behaviors,random = pars['trainingType'])
    plt.show()


#%%
###############################################    
# 
# overlay neuron projections with relevant neurons
#
##############################################
if positionweights:
   
    mp.plotWeightLocations(dataSets, resultDict, keyList, fitmethod='LASSO')
    plt.show()
#%%
###############################################    
# 
# plot the number of neurons and scatter plot of predictions fo velocity and turns
#
##############################################
if resultsPredictionOverview:
    fitmethod = 'LASSO'
    mp.plotLinearModelScatter(dataSets, resultDict, keyList, pars, fitmethod='LASSO', behaviors = ['AngleVelocity', 'Eigenworm3'], random = 'none')
    # collect the relevant number of neurons
    
    
    noNeur = []
    for key in keyList:
        noNeur.append([resultDict[key]['LASSO']['AngleVelocity']['noNeurons'], resultDict[key]['LASSO']['Eigenworm3']['noNeurons']])
    noNeur = np.array(noNeur)
    plt.figure()
    plt.bar([1,2], np.mean(noNeur, axis=0),yerr=np.std(noNeur, axis=0) )

    