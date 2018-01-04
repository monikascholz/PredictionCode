
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

# data parameters
dataPars = {'medianWindow':13, # smooth eigenworms with median filter of that size, must be odd
            'savGolayWindow':13, # savitzky-golay window for angle velocity derivative. must be odd
            'rotate':True, # rotate Eigenworms using previously calculated rotation matrix
            'savGolayWindowGCamp': 13 # savitzky-golay window for red anc green channel
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
        'trainingCut': 0.6, # what fraction of data to use for training 
        'trainingType': 'middle', # simple, random or middle.select random or consecutive data for training. Middle is a testset in the middle
        'linReg': 'simple', # ordinary or ransac least squares
        'trainingSample': 1, # take only samples that are at least n apart to have independence. 4sec = gcamp_=->24 apart
        'useRank': 0, # use the rank transformed version of neural data for all analyses
      }

behaviors = ['AngleVelocity', 'Eigenworm3', 'Eigenworm2']
#behaviors = ['AngleVelocity']

###############################################    
# 
# check which calculations to perform
#
##############################################
createIndicesTest = True 
overview = 1#False
pca = 0#False
hierclust = False
linreg = False
lasso = 1
elasticnet = 0#True
positionweights = True
###############################################    
# 
# create training and test set indices
# 
##############################################
if createIndicesTest:
    for kindex, key in enumerate(keyList):
        resultDict[key]['Training'] = {'Indices':  dr.createTrainingTestIndices(dataSets[key], pars, label='Eigenworm3')}
    print "Done generating trainingsets"
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
    mp.plotPCAresults(dataSets, resultDict, keyList, pars)
    plt.show()
    #  plot 3D trajectory of PCA
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'etho')
    plt.show()
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'velocity')
    plt.show()
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'turns')
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
    print "Performing LASSO.",
    for kindex, key in enumerate(keyList):
        print key
        trainingsInd, testInd = resultDict[key]['Training']['Indices']
        resultDict[key]['LASSO'] = dr.runLasso(dataSets[key], pars, testInd, trainingsInd, plot=1, behaviors = behaviors)
        # calculate how much more neurons contribute
        tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key], testInd, trainingsInd, pars, fitmethod = 'LASSO', behaviors = behaviors)
        for tmpKey in tmpDict.keys():
            resultDict[key]['LASSO'][tmpKey].update(tmpDict[tmpKey])
    
    mp.plotLinearModelResults(dataSets, resultDict, keyList, pars, fitmethod='LASSO', behaviors = behaviors, random = pars['trainingType'])
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
        resultDict[key]['ElasticNet'] = dr.runElasticNet(dataSets[key], pars, testInd, trainingsInd, plot=1, behaviors = behaviors)
        # calculate how much more neurons contribute
        tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key], testInd, trainingsInd,pars, fitmethod = 'ElasticNet', behaviors = behaviors, )
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
        
    mp.plotWeightLocations(dataSets, resultDict, keyList, fitmethod='LASSO')
    plt.show()