# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:34:10 2017
dimensionality reduction and linear model.
@author: monika
"""
import matplotlib.pylab as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

###############################################    
# 
# create trainings and test sets
#
############################################## 
def createTrainingTestIndices(data, pars):
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
        
    return trainingsIndices, testIndices
        

###############################################    
# 
# PCA
#
##############################################
def runPCANormal(data, pars):
    """run PCA on neural data and return nicely orgainzed dictionary."""
    nComp = pars['nCompPCA']
    pca = PCA(n_components = nComp)
    pcs = pca.fit_transform( data['Neurons']['Activity'])
    # order neurons by weight in first component
    indices = np.arange(len( data['Neurons']['Activity']))
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
        X = data['Neurons']['Activity']
#        scalerB = StandardScaler()
#        Y = scalerB.fit_transform(Y)
#        scalerN = StandardScaler()
#        X = scalerN.fit_transform(X.T).T

        tmpData = []
#        plot = False
#        nPlots = int(np.sqrt(len(data['Neurons']['Activity']))+1)
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
#            if plot:
#                #plt.subplot(nPlots, nPlots, nindex+1)
#                plt.scatter(neuron[trainingsInd], y[trainingsInd],alpha=0.1, s=1, c = 'r')
#                plt.plot(neuron[trainingsInd], neuron[trainingsInd]*reg.coef_+reg.intercept_)
#                plt.scatter(neuron[testInd], y[testInd],alpha=0.1, s=1, c='b')
#                plt.show()
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

def runLasso(data, pars, testInd, trainingsInd, plot = False):
    """run LASSO to fit behavior and neural activity with a linear model."""
    linData = {}
    for label in ['AngleVelocity', 'Eigenworm3']:
        Y = data['Behavior'][label]
        X = data['Neurons']['Activity'].T # transpose to conform to nsamples*nfeatures

        # fit lasso and validate
        reg = linear_model.LassoCV( cv=20)
        reg.fit(X[trainingsInd], Y[trainingsInd])
        scorepred = reg.score(X[testInd], Y[testInd])
        score = reg.score(X[trainingsInd], Y[trainingsInd])
        linData[label] = {}
        linData[label]['weights'] = reg.coef_
        linData[label]['intercepts'] = reg.intercept_
        linData[label]['alpha'] = reg.alpha_
        linData[label]['score'] = score
        linData[label]['scorepredicted'] = scorepred
        
        
        if plot:
            print 'alpha', reg.alpha_
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
            hist, bins = np.histogram(reg.coef_, bins = 30, density = True)
            ax1.fill_between(bins[:-1],np.zeros(len(hist)), hist, step='post', color='r')
            ax1.set_xlabel(r'weights')
            ax1.set_ylabel('PDF(weights)')
            plt.subplot(224)
            plt.plot(reg.alphas_, reg.mse_path_, 'k', alpha = 0.1)
            plt.plot(reg.alphas_, np.mean(reg.mse_path_, axis =1), 'k')
            plt.show()
    return linData
    
###############################################    
# 
# LASSO
#
##############################################    

def runElasticNet(data, pars, testInd, trainingsInd, plot = False):
    """run LASSO to fit behavior and neural activity with a linear model."""
    linData = {}
    for label in ['AngleVelocity', 'Eigenworm3']:
        Y = data['Behavior'][label]
        X = data['Neurons']['Activity'].T # transpose to conform to nsamples*nfeatures

        # fit elasticNet and validate
        l1_ratio = [.1, .5, .7, .9, .95, .99, 1]
        reg = linear_model.ElasticNetCV(l1_ratio, cv=15)
        reg.fit(X[trainingsInd], Y[trainingsInd])
        scorepred = reg.score(X[testInd], Y[testInd])
        score = reg.score(X[trainingsInd], Y[trainingsInd])
        
        #linData[label] = [reg.coef_, reg.intercept_, reg.alpha_, score, scorepred]
        linData[label] = {}
        linData[label]['weights'] = reg.coef_
        linData[label]['intercepts'] = reg.intercept_
        linData[label]['alpha'] = reg.alpha_
        linData[label]['score'] = score
        linData[label]['scorepredicted'] = scorepred
        if plot:
            print 'alpha', reg.alpha_
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
            hist, bins = np.histogram(reg.coef_, bins = 30, density = True)
            ax1.fill_between(bins[:-1],np.zeros(len(hist)), hist, step='post', color='r')
            ax1.set_xlabel(r'weights')
            ax1.set_ylabel('PDF(weights)')
            plt.subplot(224)
            for l1index, l1 in enumerate(l1_ratio):
                plt.plot(reg.alphas_[l1index], reg.mse_path_[l1index], 'k', alpha = 0.1)
                plt.plot(reg.alphas_[l1index], np.mean(reg.mse_path_[l1index], axis =1), 'k')
            plt.show()
    return linData