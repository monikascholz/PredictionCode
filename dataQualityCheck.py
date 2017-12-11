
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

# results dictionary 
resultDict = {}
for kindex, key in enumerate(keyList):
    resultDict[key] = {}
# analysis parameters

pars ={'nCompPCA':10, # no PCA components
        'PCAtimewarp':True, #timewarp so behaviors are equally represented
        'trainingCut': 0.4, # what fraction of data to use for training 
        'trainingType': 'middle', # select random or consecutive data for training. Middle is a testset in the middle
        'linReg': 'simple', # ordinary or ransac least squares
        'trainingSample': 6, # take only samples that are at least n apart to have independence
      }

###############################################    
# 
# check which calculations to perform
#
##############################################
createIndicesTest = True 
overview = False
pca = False
hierclust = False
linreg = False
lasso = True
elasticnet = False
positionweights = False
###############################################    
# 
# create training and test set indices
# 
##############################################
if createIndicesTest:
    for kindex, key in enumerate(keyList):
        resultDict[key]['Training'] = {'Indices':  dr.createTrainingTestIndices(dataSets[key], pars)}
        
###############################################    
# 
# some generic data checking plots
#
##############################################
if overview:
    mp.plotDataOverview(dataSets, keyList)
    mp.plotNeurons3D(dataSets, keyList, threed = False)  
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
        resultDict[key]['PCA'] = dr.runPCANormal(dataSets[key], pars)
    
    # overview of PCA results and weights
    mp.plotPCAresults(dataSets, resultDict, keyList)
    plt.show()
    #  plot 3D trajectory of PCA
    mp.plotPCAresults3D(dataSets, resultDict, keyList, col = 'phase')
    plt.show()
    mp.plotPCAresults3D(dataSets, resultDict, keyList, col = 'velocity')
    plt.show()
    mp.plotPCAresults3D(dataSets, resultDict, keyList, col = 'turns')
    plt.show()
#%%
###############################################    
# 
# run PCA and store results
#
##############################################
if hierclust:
    print 'run hierarchical clustering'
    print 'to implement'


#%%
###############################################    
# 
# linear regression single neurons
#
##############################################
if linreg:
    for kindex, key in enumerate(keyList):
        trainingsInd, testInd = resultDict[key]['Training']['Indices']
        resultDict[key]['Linear Regression'] = dr.linearRegressionSingleNeuron(dataSets[key], pars, testInd, trainingsInd)
    
    mp.plotLinearPredictionSingleNeurons(dataSets, resultDict, keyList)
    plt.show()
    
#%%
###############################################    
# 
# linear regression using LASSO
#
##############################################
if lasso:
    print "Performing LASSO."
    for kindex, key in enumerate(keyList):
        print key
        trainingsInd, testInd = resultDict[key]['Training']['Indices']
        resultDict[key]['LASSO'] = dr.runLasso(dataSets[key], pars, testInd, trainingsInd, plot=0)
        # calculate how much more neurons contribute
        tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key], testInd, trainingsInd, fitmethod = 'LASSO')
        for tmpKey in tmpDict.keys():
            resultDict[key]['LASSO'][tmpKey].update(tmpDict[tmpKey])
    
    mp.plotLinearModelResults(dataSets, resultDict, keyList, fitmethod='LASSO')
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
        trainingsInd, testInd = resultDict[key]['Training']['Indices']
        resultDict[key]['ElasticNet'] = dr.runElasticNet(dataSets[key], pars, testInd, trainingsInd, plot=0)
    
    mp.plotLinearModelResults(dataSets, resultDict, keyList, fitmethod='ElasticNet')
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
        
    mp.plotWeightLocations(dataSets, resultDict, keyList, fitmethod='LASSO')
    plt.show()