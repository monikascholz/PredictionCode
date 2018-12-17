# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 11:26:36 2018
Distinct information in neural traces
@author: monika
"""

import numpy as np
import matplotlib as mpl
import os
#
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.ticker as mtick
# custom pip
#import svgutils as svg
#
import makePlots as mp
import dataHandler as dh
import dimReduction as dr

from stylesheet import *
# anaysis stuff
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
#

from scipy.signal import welch, lombscargle, fftconvolve



def normalize(x):
    """zscore input"""
    return (x-np.mean(x))/np.std(x)
    

def lagCorrelation(x1, x2):
    """given two timeseries x1(t), x2(t) calculate the full lag-time crosscorrelation."""
    x1 = normalize(x1)
    x2 = normalize(x2)
    assert len(x1) == len(x2)
    periods = np.arange(-len(x1), len(x1)-1)
    return periods,fftconvolve(x1, x2[::-1], mode='full')/len(x1)

################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML32', 'AML70']:
    for condition in ['moving', 'chip']:# ['moving', 'immobilized', 'chip']:
        folder = '{}_{}/'.format(typ, condition)
        dataLog = '{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition)
        outLoc = "Analysis/{}_{}_results.hdf5".format(typ, condition)
        outLocData = "Analysis/{}_{}.hdf5".format(typ, condition)
        
        try:
            # load multiple datasets
            dataSets = dh.loadDictFromHDF(outLocData)
            keyList = np.sort(dataSets.keys())
            results = dh.loadDictFromHDF(outLoc) 
            # store in dictionary by typ and condition
            key = '{}_{}'.format(typ, condition)
            data[key] = {}
            data[key]['dsets'] = keyList
            data[key]['input'] = dataSets
            data[key]['analysis'] = results
        except IOError:
            print typ, condition , 'not found.'
            pass
print 'Done reading data.'

################################################
#
# create figure 1: This is twice the normal size
#
################################################
# we will select a 'special' dataset here, which will have all the individual plots
# select a special dataset - moving AML32. Should be the same as in fig 2
movingAML32 = 'BrainScanner20170613_134800'
moving = data['AML32_moving']['input'][movingAML32]
movingAnalysis = data['AML32_moving']['analysis'][movingAML32]

fig = plt.figure('Fig - 5 : Distinct information in neural timetraces', figsize=(9.5, 9*3/4.))
# this gridspec makes one example plot of a heatmap with its PCA
gs1 = gridspec.GridSpec(2, 6)
gs1.update(left=0.08, right=0.98, wspace=0.45, bottom = 0.1, top=0.97, hspace=0.3)
#ax1 = plt.subplot(gs1[0,0])
#ax2 = plt.subplot(gs1[0,1])
#ax3 = plt.subplot(gs1[0,2])
#ax4 = plt.subplot(gs1[1,0])
#ax5 = plt.subplot(gs1[1,1])
#ax6 = plt.subplot(gs1[1,2])
# add a,b,c letters, 9 pt final size = 18pt in this case
#letters = ['A', 'B', 'C']
#y0 = 0.99
#locations = [(0,y0),  (0.47,y0), (0.76,y0)]
#for letter, loc in zip(letters, locations):
#    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
#            horizontalalignment='left',verticalalignment='top',)
################################################
#
# first row - illustrate hierarchical clustering for one dataset
#
################################################
index = 0
flag = 'ElasticNet'
#axs = [[ax1, ax2, ax3], [ax4, ax5, ax6]]
#links = [L1, L2, L3, L0]
#set_link_color_palette(links)
tau0 = []
for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
        dset = data[key]['input']
        res = data[key]['analysis']
        for idn in dset.keys():
            neurons = dset[idn]['Neurons']['RawActivity']
            t = dset[idn]['Neurons']['Time']
            for b, (behavior, c, lbl) in enumerate(zip(['AngleVelocity', 'Eigenworm3'], [R1, B1], ['Wave speed', 'Turn'])):
                beh = dset[idn]['Behavior'][behavior]
                Weights =res[idn][flag][behavior]['weights']
                prediction =res[idn][flag][behavior]['output']
                Relevant = np.where(np.abs(Weights>0))[0]
                if len(Relevant)<1:
                    continue
                X = np.copy(dset[idn]['Neurons']['Ratio'])[Relevant]
                w = 6*60
                crosscorr = []
                for neuron in X:
                    tau, cc = lagCorrelation(beh, neuron)
                    
                    crosscorr.append(cc[int(len(cc)*0.5)-w:int(len(cc)*0.5)+w])
                if index<12:
                    ax1 = plt.subplot(gs1[b, index])
                    ax1.imshow(crosscorr, aspect='auto', extent=[-w, w, 0, len(X)])
                    
                #plt.plot(tau/6, np.mean(crosscorr, axis=0))
                #plt.show()
                #print b, tau[np.argmax(crosscorr)]
                #if idn == movingAML32:
                    #pars = None
                    #subset = Relevant
                    #clust = dr.runHierarchicalClustering(dset[idn], pars, subset)
                    
                    
#                    dn = dendrogram(clust['linkage'],ax = axs[b][0],leaf_font_size=12, leaf_rotation=45,\
#                         orientation = 'left', show_leaf_counts=1, above_threshold_color='k', color_threshold= clust['threshold'])
#                    traces = clust['clusters']
#                    for n in range( clust['nclusters']):
#                        print n
#                        axs[b][1].plot(traces[n].T+5*n, 'k', alpha=0.2)
#                        axs[b][1].plot(np.nanmean(traces[n], axis=0)+n*5, color= links[n])
#                        # sort by behavior and plot
#                        print traces[n].shape
#                        xPlot, avg, std = sortnbin(beh, np.nanmean(traces[n], axis=0), nBins=10, rng=(np.min(beh), np.max(beh)))
#                        axs[b][2].plot(xPlot, avg+n, color= links[n])
#                        axs[b][2].fill_between(xPlot, avg-std+n, avg+std+n,color= links[n], alpha=0.5)
#                        axs[b][2].plot(xPlot,n*np.ones(len(xPlot)), color= 'k', linestyle=':')
#                        
#                    axs[b][2].plot([xPlot[0], xPlot[0]], [0,0.5], color=c, lw = 2)
#                    axs[b][2].text(xPlot[0],0.25 , r'$\Delta R/ R_0 = 0.5$', verticalalignment = 'center')
#                    axs[b][2].spines['left'].set_visible(False)
#                    axs[b][2].set_xlabel(lbl)
#                    axs[b][2].set_yticks([])
#                    #ax3.plot(traces[1].T)
                    #ax3.plot(np.mean(traces[1], axis=1))
            index+=1
print tau0   
plt.show()