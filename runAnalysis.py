
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
outLoc = "AML32_moving/Analysis/Results.hdf5"
if 0:
    ##### GFP
    folder = "AML18_moving/{}_MS/"
    dataLog = "AML18_moving/AML18_datasets.txt"
    outLoc = "AML18_moving/Analysis/Results.hdf5"
    # output is stored here


# data parameters
dataPars = {'medianWindow':3, # smooth eigenworms with gauss filter of that size, must be odd
            'savGolayWindow':5, # savitzky-golay window for angle velocity derivative. must be odd
            'rotate':True, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5 # gauss window for red and green channel
            }

dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars)
keyList = np.sort(dataSets.keys())

# results dictionary 
resultDict = {}
for kindex, key in enumerate(keyList):
    resultDict[key] = {}
# analysis parameters

pars ={'nCompPCA':10, # no of PCA components
        'PCAtimewarp':True, #timewarp so behaviors are equally represented
        'trainingCut': 0.7, # what fraction of data to use for training 
        'trainingType': 'middle', # simple, random or middle.select random or consecutive data for training. Middle is a testset in the middle
        'linReg': 'simple', # ordinary or ransac least squares
        'trainingSample': 1, # take only samples that are at least n apart to have independence. 4sec = gcamp_=->24 apart
        'useRank': 0, # use the rank transformed version of neural data for all analyses
      }

behaviors = ['AngleVelocity', 'Eigenworm3']#, 'Eigenworm2', 'CMSVelocity']
#behaviors = ['AngleVelocity']

###############################################    
# 
# check which calculations to perform
#
##############################################
createIndicesTest = 1#True 
svm = 1
pca = 1#False
lasso = 1
elasticnet = 1#True
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
# run svm to predict discrete behaviors
#
##############################################
if svm:
    for kindex, key in enumerate(keyList):
        print 'running SVM'
        splits = resultDict[key]['Training']
        resultDict[key]['SVM'] = dr.discreteBehaviorPrediction(dataSets[key], pars, splits )

###############################################    
# 
# run PCA and store results
#
##############################################
#%%
if pca:
    print 'running PCA'
    for kindex, key in enumerate(keyList):
        #resultDict[key]['PCA'] = dr.runPCANormal(dataSets[key], pars)
        resultDict[key]['PCA'] = dr.runPCATimeWarp(dataSets[key], pars)


#%%
###############################################    
# 
# linear regression using LASSO
#
##############################################
if lasso:
    print "Performing LASSO.",
    for kindex, key in enumerate(keyList):
        
        splits = resultDict[key]['Training']
        resultDict[key]['LASSO'] = dr.runLasso(dataSets[key], pars, splits, plot=0, behaviors = behaviors)
        # calculate how much more neurons contribute
        tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key],splits, pars, fitmethod = 'LASSO', behaviors = behaviors)
        for tmpKey in tmpDict.keys():
            resultDict[key]['LASSO'][tmpKey].update(tmpDict[tmpKey])
        
        tmpDict = dr.reorganizeLinModel(dataSets[key], resultDict[key], splits, pars, fitmethod = 'LASSO', behaviors = behaviors)
        for tmpKey in tmpDict.keys():
            resultDict[key]['LASSO'][tmpKey]=tmpDict[tmpKey]
    
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
        resultDict[key]['ElasticNet'] = dr.runElasticNet(dataSets[key], pars,splits, plot=0, behaviors = behaviors)
        # calculate how much more neurons contribute
        tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key], splits,pars, fitmethod = 'ElasticNet', behaviors = behaviors, )
        for tmpKey in tmpDict.keys():
            resultDict[key]['ElasticNet'][tmpKey].update(tmpDict[tmpKey])
        
        tmpDict = dr.reorganizeLinModel(dataSets[key], resultDict[key], splits, pars, fitmethod = 'ElasticNet', behaviors = behaviors)
        for tmpKey in tmpDict.keys():
            resultDict[key]['ElasticNet'][tmpKey]=tmpDict[tmpKey]
    

#%%
###############################################    
# 
# save data as HDF5 file
#
##############################################
dh.saveDictToHDF(outLoc, resultDict)