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
from sklearn.metrics import explained_variance_score, accuracy_score, precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn import svm
import dataHandler as dh
from scipy.interpolate import interp1d
from scipy.optimize import newton 
# 
import makePlots as mp

###############################################    
# 
# create trainings and test sets
#
############################################## 
def splitIntoSets(y, nBins=5, nSets=5, splitMethod='auto', verbose=0):
    """get balanced training sets for a dataset y."""
    # Determine limits of array
    yLimits = np.array([np.min(y),np.max(y)])
    yLimits = np.percentile(y, [2.28, 97.72])#
    if verbose >= 1:
        print 'Min and max y: ', yLimits

    # Calculate bin edges and number of events in each bin
    nCounts,binEdges = np.histogram(y,nBins,yLimits)
    if  verbose >= 2:
        print 'Bin edges: ', binEdges
        print 'Counts: ', nCounts

    # Get minimum and maximum number of events in bins
    nCountsLimits = np.array([np.min(nCounts),np.max(nCounts)])
    if verbose >= 1:
        print 'Min and max counts: ', nCountsLimits

    # Determine bin index for each individual event
    # Digitize is semi-open, i.e. slightly increase upper limit
    binEdges[-1] += 1e-5
    binEdges[0], binEdges[-1] = np.min(y), np.max(y)+1e-5
    yBinIdx = np.digitize(y,binEdges) - 1

    # Get event indices for each bin
    eventIndices = []
    for binIdx in range(nBins):
        eventIndices.append(np.arange(len(y),dtype=int)[yBinIdx == binIdx])
    eventIndices = np.asarray(eventIndices)

    # Determine split method if auto is used
    nPerBin = nCountsLimits[0]/nSets
    if splitMethod == 'auto':
        if nPerBin < 10:
            splitMethod = 'redraw'
        else:
            splitMethod = 'unique'

    # Get proper number of events per bin, depending on split method
    if splitMethod == 'redraw':
        nPerBin = nCountsLimits[1]/nSets                            # Maximum bin count divided by number of sets
        if nPerBin > nCountsLimits[0]: nPerBin = nCountsLimits[0]   # But has to be at most the minimum bin count
    else: nPerBin = nCountsLimits[0]/nSets

    if verbose >= 1:
        print 'Split method: ', splitMethod
        print 'Events per bin per set: ', nPerBin

    # Create subsets
    sets = [[] for i in range(nSets)]
    for i in range(nBins):
        _tEvtIdx = np.asarray(eventIndices[i][:])
        for j in range(nSets):
            np.random.shuffle(_tEvtIdx)
            if len(_tEvtIdx) > nPerBin:
                sets[j].append(np.copy(_tEvtIdx[:nPerBin]))
            else:
                sets[j].append(np.copy(_tEvtIdx[:]))
            if splitMethod == 'unique':
                _tEvtIdx = _tEvtIdx[nPerBin:]

    # Convert into numpy arrays
    for i in range(nSets):
        sets[i] = np.asarray(sets[i]).reshape(nPerBin*nBins)
    sets = np.asarray(sets)

    # Prepare info dictionary with helpful data
    info = {}
    info['total-entries'] = int(sets.shape[0] * sets.shape[1])
    info['unique-entries'] = len(list(set(np.ravel(sets))))
    info['method'] = splitMethod
    info['min-max-y'] = yLimits
    info['min-max-cts'] = nCountsLimits
    info['nbins'] = nBins

    return sets, info
    
def createTrainingTestIndices(data, pars, label):
    """split time points into trainings and test set."""
    timeLen = data['Neurons']['Activity'].shape[1]
    if pars['trainingType'] == 'start':
        cutoff = int(pars['trainingCut']*timeLen)
        testIndices = np.arange(timeLen)[:cutoff:]
        trainingsIndices = np.arange(timeLen)[cutoff::pars['trainingSample']]
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
        #center = np.where(np.abs(data['Behavior']['Eigenworm3'])>15)[0]
        cutoff = int((pars['trainingCut'])*timeLen/2.)
        
        tmpIndices = np.arange(timeLen)
        testIndices = tmpIndices[cutoff:-cutoff]
        # create a trainingset by equalizing probabilities
        trainingsIndices = np.setdiff1d(tmpIndices, testIndices)
        # bin  to get probability distribution
        nbin = 5
        hist, bins = np.histogram(data['Behavior'][label], nbin)
        # this is the amount of data that will be left in each bin after equalization
        #N = np.sum(hist)/20.#hist[0]+hist[-1]
        N = np.max(hist)/2#*nbin
        print bins, np.min(data['Behavior'][label]), np.max(data['Behavior'][label])
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
    #Neuro = np.array([dh.savitzky_golay(line, window_size=15, order=3, deriv=1) for line in Neuro])
    pcs = pca.fit_transform(Neuro)
    comp = pca.components_
    #comp = np.cumsum(pca.components_, axis=1)
    # order neurons by weight in first component
    #indices = np.arange(len(Neuro))
    
    indices = np.argsort(pcs.T)[0]
    #print indices.shape
    pcares = {}
    pcares['nComp'] =  pars['nCompPCA']
    pcares['expVariance'] =  pca.explained_variance_ratio_
    pcares['neuronWeights'] =  pcs
    pcares['neuronOrderPCA'] =  indices
    pcares['pcaComponents'] =  comp
    
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
    # number of timestamps in each behavior
    lens = np.array([len(x) for x in indices])
    # timesamples in the smallest behavior
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
# predict discrete behaviors
#
##############################################
    
def discreteBehaviorPrediction(data, pars, splits):
    """use a svm to predict discrete behaviors from the data."""
    # modify data to be like scikit likes it
    if pars['useRank']:
            X = data['Neurons']['rankActivity'].T
    else:
        X = data['Neurons']['Activity'].T # transpose to conform to nsamples*nfeatures
    # use ethogram for behavior
    Y = data['Behavior']['Ethogram']
    trainingsInd, testInd = splits['AngleVelocity']['Indices']
    # create a linear SVC
    lin_clf = svm.LinearSVC(penalty='l1',dual=False, class_weight='balanced', C=10)
    
    lin_clf.fit(X[trainingsInd], Y[trainingsInd]) 
    
    Ypred = lin_clf.predict(X[testInd])
    
    print classification_report(Y[testInd], Ypred)
    pcs = lin_clf.coef_
    indices = np.argsort(pcs[0])
    
    comp = np.zeros((4, len(X)))
    # show temporal components
    for wi, weights in enumerate(pcs):
        comp[wi] = np.dot(X, weights)
    #print f1_score(Y[testInd], Ypred, average='micro')
    recision, recall, fscore, support = precision_recall_fscore_support(Y[testInd], Ypred, labels=[-1,0,1,2])
    print fscore
    pcares = {}
    pcares['nComp'] =  4
    pcares['expVariance'] =  fscore
    pcares['neuronWeights'] =  pcs.T
    pcares['neuronOrderPCA'] =  indices
    pcares['pcaComponents'] =  comp
#    X_train = X[trainingsInd]
#    y_train = Y[trainingsInd]
#    X_test = X[testInd]
#    y_test = Y[testInd]
#    # Set the parameters by cross-validation
#    tuned_parameters = [ {'kernel': ['linear'], 'C': [1, 5,10,100]}]
#    
#    scores = ['f1_weighted', 'f1_macro']
#    
#    for score in scores:
#        print("# Tuning hyper-parameters for %s" % score)
#        print()
#    
#        clf = GridSearchCV(svm.SVC(class_weight='balanced'), tuned_parameters, cv=5,
#                           scoring='%s' % score)
#        clf.fit(X_train, y_train)
#    
#        print("Best parameters set found on development set:")
#        
#        print(clf.best_params_)
#        
#        print("Grid scores on development set:")
#       
#        means = clf.cv_results_['mean_test_score']
#        stds = clf.cv_results_['std_test_score']
#        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#            print("%0.3f (+/-%0.03f) for %r"
#                  % (mean, std * 2, params))
#        print()
#    
#        print("Detailed classification report:")
#        print()
#        print("The model is trained on the full development set.")
#        print("The scores are computed on the full evaluation set.")
#        print()
#        y_true, y_pred = y_test, clf.predict(X_test)
#        print(classification_report(y_true, y_pred))
#        print()
    T = testInd
    ax1 = plt.subplot(211)
    mp.plotEthogram(ax1, T, Y[testInd], alpha = 0.5, yValMax=1, yValMin=0, legend=0)
    ax1 = plt.subplot(212)
    mp.plotEthogram(ax1, T, Ypred, alpha = 0.5, yValMax=1, yValMin=0, legend=0)
    return pcares
###############################################    
# 
# LASSO
#
##############################################    

def stdevRule(x, y, std):
    """move by one stdeviation to increase regularization."""
    yUp = np.min(y) + std[np.argmin(y)]
    yFunc =  interp1d(x,y,'cubic',fill_value='extrapolate')
#    plt.plot(x,y)
#    plt.errorbar(x,y, yerr=std)
#    plt.show()
#    y0, y1 = np.min(y), y[-1]
#    x0, x1 = x[np.argmin(y)], x[-1]
#    m = (y1-y0)/(x1-x0)
#    c = y0-m*x0
#    xalpha = (c-yUp)/m
    xalpha = x[np.argmin(y)]*1.5
    xUpper = newton(lambda x: np.abs(yFunc(x) - yUp), xalpha)
#    print xUpper, xalpha
#    plt.plot(np.abs(yFunc(x) - yUp))
#    
#    plt.show()
    return xUpper
def balancedFolds(y):
    """create balanced train/validate splitsby leave one out."""
    splits, _ = splitIntoSets(y, nBins=5, nSets=10, splitMethod='unique', verbose=0)
    folds = []
    for i in range(len(splits)):
        folds.append([splits[i], np.concatenate(splits[np.arange(len(splits))!=i] )])
    return folds
    
def runLasso(data, pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3']):
    """run LASSO to fit behavior and neural activity with a linear model."""
    linData = {}
    for label in behaviors:
        Y = data['Behavior'][label]
        trainingsInd, testInd = splits[label]['Indices']
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
        a = np.logspace(-3,1,100)
        cv = 15
        # unbalanced sets
        #fold = KFold(cv, shuffle=False) 
        # balanced sets
        fold = balancedFolds(Y[trainingsInd])
        reg = linear_model.LassoCV(cv=fold, selection = 'random', verbose=0, \
        eps=1e-1, max_iter=3000, alphas=a)#, normalize=False)
        reg.fit(X[trainingsInd], Y[trainingsInd])
        alphas = reg.alphas_
        ymean = np.mean(reg.mse_path_, axis =1)
        yerr = np.std(reg.mse_path_, axis =1)/np.sqrt(cv)
        alphaNew = alphas[np.argmin(ymean)]
#        # calculate standard deviation rule
#        alphaNew = stdevRule(x = alphas, y= ymean, std= yerr)
#        reg = linear_model.Lasso(alpha=alphaNew)
#        reg.fit(X[trainingsInd], Y[trainingsInd])
        
        
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
            
            #plt.plot(reg.alphas_, reg.mse_path_, 'k', alpha = 0.3)
            plt.plot(alphas, ymean, 'k')
            plt.errorbar(alphas, ymean, yerr=yerr, capsize=1)
            plt.axvline(alphas[np.argmin(ymean)],label='minimal error')
            plt.axvline(alphaNew,label='stdev rule')
            plt.xscale('log')
            #plt.fill_between(reg.alphas_,y1=ymean-yerr, y2= ymean+yerr, alpha=0.5)
            #plt.errorbar(,color= 'k')
            plt.tight_layout()            
            plt.show()
        # score model
            
        scorepred = reg.score(X[testInd], Y[testInd])#, sample_weight=np.power(Y[testInd], 2))
        score = reg.score(X[trainingsInd], Y[trainingsInd])
        linData[label] = {}
        linData[label]['weights'] =  reg.coef_
        linData[label]['intercepts'] = reg.intercept_
        linData[label]['alpha'] = alphaNew#reg.alpha_
        linData[label]['score'] = score
        linData[label]['scorepredicted'] = scorepred
        linData[label]['noNeurons'] = len(reg.coef_[np.abs(reg.coef_)>0])
        print 'alpha', alphaNew, scorepred
    return linData
    
###############################################    
# 
# ElasticNet
#
##############################################    

def runElasticNet(data, pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3']):
    """run EN to fit behavior and neural activity with a linear model."""
    linData = {}
    for label in behaviors:
        Y = data['Behavior'][label]
        #Y = preprocessing.scale(Y)
        if pars['useRank']:
            X = data['Neurons']['rankActivity'].T
        else:
            X = data['Neurons']['Activity'].T # transpose to conform to nsamples*nfeatures
        trainingsInd, testInd = splits[label]['Indices']
        # fit elasticNet and validate
        l1_ratio = [0.5, 0.75, .9, .95, 0.99, 1]
        #cv = 15
        a = np.linspace(0.0001,0.05,100)
        fold = balancedFolds(Y[trainingsInd])
        reg = linear_model.ElasticNetCV(l1_ratio, cv=fold, verbose=0)
        #reg = linear_model.ElasticNetCV(cv=cv, verbose=1)
        reg.fit(X[trainingsInd], Y[trainingsInd])
        scorepred = reg.score(X[testInd], Y[testInd])
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
            # use if only one l1_ratio
            
#            plt.plot(reg.alphas_, reg.mse_path_, 'k', alpha = 0.1)
#            plt.plot(reg.alphas_, np.mean(reg.mse_path_, axis =1))
#            plt.show()
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
def scoreModelProgression(data, results, splits, pars, fitmethod = 'LASSO', behaviors = ['AngleVelocity', 'Eigenworm3']):
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
        trainingsInd, testInd = splits[label]['Indices']
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
    
###############################################    
# 
# reorganize linear models to fit PCA plot style
#
##############################################
def reorganizeLinModel(data, results, splits, pars, fitmethod = 'LASSO', behaviors = ['AngleVelocity', 'Eigenworm3']):
    """takes a model fit and calculates basis vectors etc."""
    # modify data to be like scikit likes it
    if pars['useRank']:
        X = data['Neurons']['rankActivity'].T
    else:
        X = data['Neurons']['Activity'].T # transpose to conform to nsamples*nfeatures
    # get weights
    pcs = np.vstack([results[fitmethod][label]['weights'] for label in behaviors])
    print pcs.shape
    indices = np.argsort(pcs[0])
    
    comp = np.zeros((len(behaviors), len(X)))
    # show temporal components
    for wi, weights in enumerate(pcs):
        print weights.shape
        comp[wi] = np.dot(X, weights)
    # calculate redicted neural dynamics  TODO
    score = np.ones(len(behaviors))
    #score = explained_variance_score()
    
    pcares = {}
    pcares['nComp'] =  len(behaviors)
    pcares['expVariance'] =  score
    pcares['neuronWeights'] =  pcs.T
    pcares['neuronOrderPCA'] =  indices
    pcares['pcaComponents'] =  comp
    return pcares
