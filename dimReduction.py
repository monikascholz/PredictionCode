# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:34:10 2017
dimensionality reduction and linear model.
@author: monika
"""
import matplotlib.pylab as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.cluster import AgglomerativeClustering
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing

###############################################    
# 
# create trainings and test sets
#
############################################## 
def createTrainingTestIndices(data, pars, label):
    """split time points into trainings and test set."""
    timeLen = data['Neurons']['Activity'].shape[1]
    if pars['trainingType'] == 'simple':
        cutoff = int(pars['trainingCut']*timeLen)
        trainingsIndices = np.arange(timeLen)[:cutoff:pars['trainingSample']]
        testIndices = np.arange(timeLen)[cutoff:]
    elif pars['trainingType'] == 'random':
        cutoff = int(pars['trainingCut']*timeLen)
        tmpindices = np.arange(timeLen)
        np.random.shuffle(tmpindices[::pars['trainingSample']])
        trainingsIndices = np.sort(tmpindices[:cutoff])
        testIndices = np.sort(tmpindices[cutoff:])
    elif pars['trainingType'] == 'middle':
        cutoff = int((pars['trainingCut'])*timeLen/2.)
        tmpIndices = np.arange(timeLen)
        testIndices = tmpIndices[cutoff:-cutoff]
        trainingsIndices = np.setdiff1d(tmpIndices, testIndices)[::pars['trainingSample']]
    elif pars['trainingType'] == 'LR':
        # crop out a testset first -- find an area that contains at least one turn
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
# create behavioral signatures
#
##############################################
def behaviorTAvg(data, pars):
    """use ethogram to calculate behavior triggered averages."""
    

###############################################    
# 
# PCA
#
##############################################

def runPCANormal(data, pars):
    """run PCA on neural data and return nicely organized dictionary."""
    nComp = pars['nCompPCA']
    pca = PCA(n_components = nComp)
    if pars['useRank']:
        Neuro = data['Neurons']['rankActivity']
    else:
        Neuro = data['Neurons']['Activity']
    pcs = pca.fit_transform(Neuro )
    # order neurons by weight in first component
    #indices = np.arange(len(Neuro))
    indices = np.argsort(pcs[:,0])
    pcares = {}
    pcares['nComp'] =  pars['nCompPCA']
    pcares['expVariance'] =  pca.explained_variance_ratio_
    pcares['neuronWeights'] =  pcs
    pcares['neuronOrderPCA'] =  indices
    pcares['pcaComponents'] =  pca.components_
    
    return pcares
    
def runPCATimeWarp(data, pars):
    """run PCA on neural data and return nicely orgainzed dictionary."""
    nComp = pars['nCompPCA']
    pca = PCA(n_components = nComp)
    neurons = timewarp(data)
    pcs = pca.fit_transform(neurons.T)
    
    pcs = pca.transform(data['Neurons']['Activity'].T)
    
    # order neurons by weight in first component
    indices = np.arange(len( data['Neurons']['Activity']))
    indices = np.argsort(pca.components_[0])
    
    pcares = {}
    pcares['nComp'] =  pars['nCompPCA']
    pcares['expVariance'] =  pca.explained_variance_ratio_
    pcares['neuronWeights'] =  pca.components_.T
    pcares['neuronOrderPCA'] =  indices
    pcares['pcaComponents'] =  pcs.T
    
    return pcares

def timewarp(data):
    """creates a subsampled neuron signal for PCA that has equally represented behaviors."""
    # find out how much fwd and backward etc we have:
    #labelDict = {-1:'Reverse',0:'Pause',1:'Forward',2:'Turn'}
    neur = data['Neurons']['Activity']
    # find how often each behavior occurs
    indices = []
    for bindex, behavior in enumerate([-1,1,2]):
        # find behavior indices
        indices.append(np.where(data['Behavior']['Ethogram']==behavior)[0])
    lens = np.array([len(x) for x in indices])
    minval = np.min(lens[np.nonzero(lens)])
    #subsample neural data to the minimal, non-zero size
    neurArr = []
    for i in range(3):
        
        if lens[i] !=0:
#            # this subsamples
#            useOnly = np.arange(0,lens[i], np.int(lens[i]/minval))[:minval]
#            tmp = neur[:,indices[i][useOnly]]
#            neurArr.append(tmp)
            # this averages
            tmp = neur[:,indices[i]]
            end =  minval * int(lens[i]/minval)
            neurArr.append(np.mean(tmp[:,:end].reshape(tmp.shape[0],-1, minval), 1))
            
    neurData = np.concatenate(neurArr, axis=1)
    return neurData
    
###############################################    
# 
# hierarchical clustering
#
##############################################    

def runHierarchicalClustering(data, pars):
    """cluster neural data."""
    


###############################################    
# 
# Linear regression individual neurons
#
##############################################  
def linearRegressionSingleNeuron(data, pars, testInd, trainingsInd):
    """for each neuron do linear regression to behavior and test predictive power."""
    if pars['linReg']=='simple':
        reg = linear_model.LinearRegression()
    elif pars['linReg']=='ransac':
        reg = linear_model.RANSACRegressor()
    else:
        return ValueError
    linData = {}
    for label in ['AngleVelocity', 'Eigenworm3']:
        Y = data['Behavior'][label]
        if pars['useRank']:
            X = data['Neurons']['rankActivity']
        else:
            X = data['Neurons']['Activity']
#        scalerB = StandardScaler()
#        Y = scalerB.fit_transform(Y)
#        scalerN = StandardScaler()
#        X = scalerN.fit_transform(X.T).T

        tmpData = []
        plot = False
        nPlots = int(np.sqrt(len(data['Neurons']['Activity']))+1)
        for nindex, neuron in enumerate(X):
            neuron = np.reshape(neuron, (-1,1))
            reg.fit(neuron[trainingsInd], Y[trainingsInd])
            #
            #reg.predict(neuron[testInd])
            scorepred = reg.score(neuron[testInd], Y[testInd])
            score = reg.score(neuron[trainingsInd], Y[trainingsInd])
            # rescale coefficients to match real numbers
#            slope_scaled = reg.coef_*scalerB.std_/scalerN.std_[nindex]
#            intercept_scaled =(-scalerN.mean_[nindex]/scalerN.std_[nindex]*reg.coef_ + scalerB.mean_/scalerB.std_+reg.intercept_)*scalerB.std_
            tmpData.append(np.array([reg.coef_, reg.intercept_, score, scorepred]))
#            
#            # plot all regressions
            if plot:
                xfit = np.arange(np.min(neuron), np.max(neuron))
                plt.subplot(nPlots, nPlots, nindex+1)
                plt.scatter(neuron[trainingsInd], Y[trainingsInd],alpha=0.1, s=1, c = 'r')
                plt.plot(xfit,xfit*reg.coef_+reg.intercept_)
                plt.scatter(neuron[testInd], Y[testInd],alpha=0.1, s=1, c='b')
        plt.show()
        tmpData = np.array(tmpData)
        coef_, intercept_, score, scorepred = tmpData.T
        linData[label] = {}
        linData[label]['weights'] = coef_
        linData[label]['intercepts'] = intercept_
        linData[label]['alpha'] = None
        linData[label]['score'] = score
        linData[label]['scorepredicted'] = scorepred
        #linData[label] = tmpData.T
    return linData
            
###############################################    
# 
# LASSO
#
##############################################    

def runLasso(data, pars, testInd, trainingsInd, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3']):
    """run LASSO to fit behavior and neural activity with a linear model."""
    linData = {}
    for label in behaviors:
        Y = data['Behavior'][label]
        #Y = Y.reshape((-1,1))
        #scale = np.nanmean(Y)
        #Y-=scale
        # regularize behavior
        #Y = preprocessing.scale(Y)
        #scaler = StandardScaler(with_std=False)
        #Y = scaler.fit_transform(Y)
        if pars['useRank']:
            X = data['Neurons']['rankActivity'].T
        else:
            X = data['Neurons']['Activity'].T # transpose to conform to nsamples*nfeatures

        # fit lasso and validate
        a = np.linspace(0.001,0.05,100)
        reg = linear_model.LassoCV(cv=5, selection = 'random', verbose=1, \
        eps=0.001, max_iter=3000, alphas=a)#, normalize=False)
        reg.fit(X[trainingsInd], Y[trainingsInd])
        
        # TODO use one-stdev rule: all else equal, regularize more
        #mean, stdev = np.mean(reg.mse_path_, axis =1), np.stdev(reg.mse_path_, axis =1)
        #alpha = np.interp(reg.alphas_, mean+stdev)
        if plot:
            plt.subplot(221)
            plt.title('Trainingsset')
            plt.plot(Y[trainingsInd], 'r')
            plt.plot(reg.predict(X[trainingsInd]), 'k', alpha=0.7)
            plt.subplot(222)
            plt.title('Testset')
            plt.plot(Y[testInd], 'r')
            plt.plot(reg.predict(X[testInd]), 'k', alpha=0.7)
            ax1 = plt.subplot(223)
            plt.title('Non-zero weighths: {}'.format(len(reg.coef_[reg.coef_!=0])))
            ax1.scatter(Y[testInd], reg.predict(X[testInd]), alpha=0.7, s=0.2)
#            hist, bins = np.histogram(reg.coef_, bins = 30, density = True)
#            ax1.fill_between(bins[:-1],np.zeros(len(hist)), hist, step='post', color='r')
#            ax1.set_xlabel(r'weights')
#            ax1.set_ylabel('PDF(weights)')
            plt.subplot(224)
            plt.plot(reg.alphas_, reg.mse_path_, 'k', alpha = 0.3)
            plt.plot(reg.alphas_, np.mean(reg.mse_path_, axis =1), 'k')
            ymean = np.mean(reg.mse_path_, axis =1)
            yerr = np.std(reg.mse_path_, axis =1)
            
            plt.fill_between(reg.alphas_,y1=ymean-yerr, y2= ymean+yerr, alpha=0.5)
            #plt.errorbar(,color= 'k')
            plt.tight_layout()            
            plt.show()
        # score model
            
        scorepred = reg.score(X[testInd], Y[testInd])#, sample_weight=np.power(Y[testInd], 2))
        score = reg.score(X[trainingsInd], Y[trainingsInd])
        linData[label] = {}
        linData[label]['weights'] =  reg.coef_
        linData[label]['intercepts'] = reg.intercept_
        linData[label]['alpha'] = reg.alpha_
        linData[label]['score'] = score
        linData[label]['scorepredicted'] = scorepred
        linData[label]['noNeurons'] = len(reg.coef_[np.abs(reg.coef_)>0])
        print 'alpha', reg.alpha_, scorepred
    return linData
    
###############################################    
# 
# ElasticNet
#
##############################################    

def runElasticNet(data, pars, testInd, trainingsInd, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3']):
    """run EN to fit behavior and neural activity with a linear model."""
    linData = {}
    for label in behaviors:
        Y = data['Behavior'][label]
        #Y = preprocessing.scale(Y)
        if pars['useRank']:
            X = data['Neurons']['rankActivity'].T
        else:
            X = data['Neurons']['Activity'].T # transpose to conform to nsamples*nfeatures

        # fit elasticNet and validate
        l1_ratio = [0.5, .9, .95, 0.99, 1]
        cv = 10
        a = np.linspace(0.0001,0.05,100)
        reg = linear_model.ElasticNetCV(l1_ratio, cv=cv, verbose=1)
        #reg = linear_model.ElasticNetCV(cv=cv, verbose=1)
        reg.fit(X[trainingsInd], Y[trainingsInd])
        scorepred = reg.score(X[testInd], Y[testInd], sample_weight=np.abs(Y[testInd]))
        score = reg.score(X[trainingsInd], Y[trainingsInd])
        
        #linData[label] = [reg.coef_, reg.intercept_, reg.alpha_, score, scorepred]
        linData[label] = {}
        linData[label]['weights'] = reg.coef_
        linData[label]['intercepts'] = reg.intercept_
        linData[label]['alpha'] = reg.alpha_
        linData[label]['l1_ratio'] = reg.l1_ratio_
        linData[label]['score'] = score
        linData[label]['scorepredicted'] = scorepred
        linData[label]['noNeurons'] = len(reg.coef_[reg.coef_>0])
        if plot:
            print 'alpha', reg.alpha_, 'l1_ratio', reg.l1_ratio_, 'r2', scorepred
            print reg.alphas_.shape, reg.mse_path_.shape
            plt.subplot(221)
            plt.title('Trainingsset')
            plt.plot(Y[trainingsInd], 'r')
            plt.plot(reg.predict(X[trainingsInd]), 'k', alpha=0.7)
            plt.subplot(222)
            plt.title('Testset')
            plt.plot(Y[testInd], 'r')
            plt.plot(reg.predict(X[testInd]), 'k', alpha=0.7)
            ax1 = plt.subplot(223)
            plt.title('Non-zero weighths: {}'.format(len(reg.coef_[reg.coef_!=0])))
            ax1.scatter(Y[testInd], reg.predict(X[testInd]), alpha=0.7, s=0.2)
            #hist, bins = np.histogram(reg.coef_, bins = 30, density = True)
            #ax1.fill_between(bins[:-1],np.zeros(len(hist)), hist, step='post', color='r')
            #ax1.set_xlabel(r'weights')
            #ax1.set_ylabel('PDF(weights)')
            plt.subplot(224)
#            
            #for l1index, l1 in enumerate(l1_ratio):
            #plt.plot(reg.alphas_, reg.mse_path_, 'k', alpha = 0.1)
            #plt.plot(reg.alphas_, np.mean(reg.mse_path_, axis =1))
            #plt.show()
            for l1index, l1 in enumerate(l1_ratio):
                plt.plot(reg.alphas_[l1index], reg.mse_path_[l1index], 'k', alpha = 0.1)
                plt.plot(reg.alphas_[l1index], np.mean(reg.mse_path_[l1index], axis =1))
            plt.show()
    return linData

###############################################    
# 
# Show how prediction improves with more neurons
#
##############################################  
def scoreModelProgression(data, results, testInd, trainingsInd, pars, fitmethod = 'LASSO', behaviors = ['AngleVelocity', 'Eigenworm3']):
    """show how more neurons improve predictive abilities."""
    linData = {}
    for label in behaviors:
        # get the weights from previously fit data and sort by absolute amplitude
        weights = results[fitmethod][label]['weights']
        weightsInd = np.argsort(np.abs(weights))[::-1]
        
        # sort neurons by weight
        Y = data['Behavior'][label]
        if pars['useRank']:
            X = data['Neurons']['rankActivity'].T
        else:
            X = data['Neurons']['Activity'].T # transpose to conform to nsamples*nfeatures
        # individual predictive scores
        indScore = []
        sumScore = []
        if fitmethod == 'ElasticNet':
            print results[fitmethod][label]['alpha'], results[fitmethod][label]['l1_ratio']
        # TODO: check why individual fits not as stable
        for count, wInd in enumerate(weightsInd):
            if np.abs(weights[wInd]) >0:
                # fit one neuron
                if fitmethod == 'LASSO':
                    reg = linear_model.Lasso(alpha = results[fitmethod][label]['alpha'])
                elif fitmethod == 'ElasticNet':
                    
                    reg = linear_model.ElasticNet(alpha = results[fitmethod][label]['alpha'],
                                                  l1_ratio = results[fitmethod][label]['l1_ratio'])
                xTmp = np.reshape(X[:,wInd], (-1,1))
                reg.fit(xTmp[trainingsInd], Y[trainingsInd])
                indScore.append(reg.score(xTmp[testInd], Y[testInd]))
#                plt.scatter(xTmp[testInd], Y[testInd], s=0.5)
#                plt.plot(xTmp[testInd], reg.predict(xTmp[testInd]))
#                plt.show()
                # fit up to n neurons
                #reg = linear_model.Lasso(alpha = results[fitmethod][label]['alpha'])
                xTmp = np.reshape(X[:,weightsInd[:count+1]], (-1,count+1))
                reg.fit(xTmp[trainingsInd], Y[trainingsInd])
                
                sumScore.append(reg.score(xTmp[testInd], Y[testInd]))
#                plt.subplot(311)
#                plt.plot(Y[trainingsInd])
#                plt.plot(reg.predict(xTmp[trainingsInd]))
#                
#                plt.subplot(312)
#                plt.plot(Y[testInd])
#                plt.plot(reg.predict(xTmp[testInd]), lw=0.5)
#                
#                plt.subplot(313)
#                plt.scatter(reg.predict(xTmp[testInd]),Y[testInd], s=0.2)
                
            plt.show()
        linData[label] = {}
        linData[label]['cumulativeScore'] = sumScore
        linData[label]['individualScore'] = indScore
    return linData
    
    
