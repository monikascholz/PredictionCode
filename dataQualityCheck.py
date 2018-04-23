
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
typ='AML70'

# GCamp6s; lite-1
if typ =='AML70': 
    folder = "AML70_moving/{}_MS/"
    dataLog = "AML70_moving/AML70_datasets.txt"
    outLoc = "AML70_moving/Analysis/"
# GCamp6s 
if typ =='AML32': 
    folder = "AML32_moving/{}_MS/"
    dataLog = "AML32_moving/AML32_datasets.txt"
    outLoc = "AML32_moving/Analysis/"
##### GFP
elif typ =='AML18': 
    folder = "AML18_moving/{}_MS/"
    dataLog = "AML18_moving/AML18_datasets.txt"
    outLoc = "AML18_moving/Analysis/"
# immobile GCamp6

elif typ =='AML32imm': 
    folder = "AML32_immobilized/{}_MS/"
    dataLog = "AML32_immobilized/AML32_immobilized_datasets.txt"

# data parameters
dataPars = {'medianWindow':5, # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow':10, # sgauss window for angle velocity derivative. must be odd
            'rotate':True, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5  # gauss window for red and green channel
            }


dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars, nDatasets = 4)
keyListAll = np.sort(dataSets.keys())
print keyListAll
for key in keyListAll: 
    keyList = keyListAll
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
            'useDeconv': 0, # use the rank transformed version of neural data for all analyses
          }
    
    behaviors = ['AngleVelocity','Eigenworm3']#, 'Eigenworm2']
    #behaviors = ['Eigenworm3']

    ###############################################    
    # 
    # check which calculations to perform
    #
    ##############################################
    createIndicesTest = 1#True 
    overview = 0#False
    bta = 0
    svm = 0
    pca = 0#False
    hierclust = False
    linreg = False
    lasso = 0
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
    # use behavior triggered averaging to create non-othogonal axes
    #
    ##############################################
    if bta:
        for kindex, key in enumerate(keyList):
            print 'running BTA'
            
            resultDict[key]['BTA'] =dr.runBehaviorTriggeredAverage(dataSets[key], pars)
        mp.plotPCAresults(dataSets, resultDict, keyList, pars,  flag = 'BTA')
        plt.show()
        mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'etho', flag = 'BTA')
        plt.show()
        ###############################################    
    # 
    # use svm to predict discrete behaviors
    #
    ##############################################
    if svm:
        for kindex, key in enumerate(keyList):
            print 'running SVM'
            splits = resultDict[key]['Training']
            resultDict[key]['SVM'] = dr.discreteBehaviorPrediction(dataSets[key], pars, splits )
        
        
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
        print 'running PCA'
        for kindex, key in enumerate(keyList):
            if typ=='AML32imm':
                print 'running normal pca'
                resultDict[key]['PCA'] = dr.runPCANormal(dataSets[key], pars)
            else:
                resultDict[key]['PCA'] = dr.runPCATimeWarp(dataSets[key], pars)
        
        # overview of data ordered by PCA
        mp.plotDataOverview2(dataSets, keyList, resultDict)
        # overview of PCA results and weights
        mp.plotPCAresults(dataSets, resultDict, keyList, pars)
        plt.show()
        mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'time')
        plt.show()
        if typ!='AML32imm':
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
            splits = resultDict[key]['Training']
            resultDict[key]['Linear Regression'] = dr.linearRegressionSingleNeuron(dataSets[key], pars, splits)
        
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
    