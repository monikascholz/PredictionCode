# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 08:24:57 2018
fig1-S3 - estimate PCA noise level and covariances from reconstruction.
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
from scipy.stats import ttest_ind
import dataHandler as dh
# deliberate import all!
from stylesheet import *
################################################
#
# create figure 1: This is twice the normal size
#
################################################
fig = plt.figure('Fig1 - S3 : PCA noise floor and cov reconstructions.', figsize=(9.5, 9*1/4.))
gs1 = gridspec.GridSpec(1,4,  width_ratios=[1, 1,1,1])
gs1.update(left=0.055, right=0.95,  bottom = 0.25, top=0.9, hspace=0.1, wspace=0.45)
fig.patch.set_alpha(0.0)
#eigenvalue axes
ax1 = plt.subplot(gs1[0,0])
ax2 = plt.subplot(gs1[0,1])
ax3 = plt.subplot(gs1[0,2])
ax4 = plt.subplot(gs1[0,3])
##covariance axes
#ax11 = plt.subplot(gs1[1,0])
#ax21 = plt.subplot(gs1[1,1])
#ax31 = plt.subplot(gs1[1,2])
#ax41 = plt.subplot(gs1[1,3])
# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B', 'C', 'D']
y0 = 0.92
locations = [(0, y0), (0.24, y0), (0.49,y0), (0.72, y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML32', 'AML18', 'AML70', 'AML175']:
    for condition in ['chip', 'moving', 'immobilized']:# ['moving', 'immobilized', 'chip']:
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


#################################################
##
## Show eigenvalues from PCA
##
#################################################
# variance explained for moving and immobile 
nComp =10
movExp = ['AML32_moving', 'AML70_chip']
imExp = ['AML32_immobilized', 'AML70_immobilized']
movCtrl = ['AML18_moving', 'AML175_moving']
imCtrl = ['AML18_immobilized']

for condition, keys, ax in zip([ 'immobilized','moving', 'immobilized (Ctrl)','moving (Ctrl)'], [ imExp,movExp,imCtrl, movCtrl ],  [ax1, ax2, ax3, ax4]):
    tmpdata = []
    noiseS = []
    noiseL = []    
    for key in keys:
        dset = data[key]['analysis']
        for idn in dset.keys():
            results=  dset[idn]['PCA']
            
            tmpdata.append(results['eigenvalue'][:nComp])
            noiseS.append(results['fullShuffle'][:nComp])
            noiseL.append(results['lagShuffle'][:nComp])
          
        #ax1.plot(np.arange(1,nComp+1),np.array(tmpdata).T ,'-',color =colorsExp[condition], lw=1, label = '{} {}'.format(typ, condition),alpha=0.3 )
    
    ax.errorbar(np.arange(1,nComp+1), np.mean(noiseS, axis=0), np.std(noiseS, axis=0), color = 'k', marker='x')
    ax.errorbar(np.arange(1,nComp+1), np.mean(noiseL, axis=0), np.std(noiseL, axis=0), color = 'b', marker='o')
    ax.set_ylabel('Eigenvalues')
    ax.errorbar(np.arange(1,nComp+1), np.mean(tmpdata, axis=0), np.std(tmpdata, axis=0), color = 'r', marker='s')    
    ax.set_title(condition)
    x0 = np.arange(1,nComp+1)[np.where((np.mean(tmpdata, axis=0)-np.mean(noiseL, axis=0))<0)[0][0]]
    
    t, p = ttest_ind(tmpdata, noiseL, axis=0, equal_var=False)
    x0 = np.where(p>0.05)[0][0]
    print x0, len(noiseL)
    ax.axvline(x0, color='k', linestyle='--')
    #ax.set_yticks([0,25,50,75,100])
    #ax12.set_yticks([0,25,50,75,100])
    ax.set_xlabel('# of components')

plt.show()


#################################################
##
## Show covariance matrices
##
#################################################
# variance explained for moving and immobile 
nComp =10
movExp = ['AML32_moving', 'AML70_chip']
imExp = ['AML32_immobilized', 'AML70_immobilized']
movCtrl = ['AML18_moving', 'AML175_moving']
imCtrl = ['AML18_immobilized']

for condition, keys, ax in zip([ 'immobilized','moving', 'immobilized (Ctrl)','moving (Ctrl)'], [ imExp,movExp,imCtrl, movCtrl ],  [ax1, ax2, ax3, ax4]):
    for key in keys:
        dset = data[key]['analysis']
        tmpdata = []
        noiseS = []
        noiseL = []
        for idn in dset.keys():
            results=  dset[idn]['PCA']
            tmpdata.append(results['eigenvalue'][:nComp])
            noiseS.append(results['fullShuffle'][:nComp])
            noiseL.append(results['lagShuffle'][:nComp])
          
    ax.errorbar(np.arange(1,nComp+1), np.mean(noiseS, axis=0), np.std(noiseS, axis=0), color = 'k', marker='x')
    ax.errorbar(np.arange(1,nComp+1), np.mean(noiseL, axis=0), np.std(noiseL, axis=0), color = 'b', marker='o')
    ax.set_ylabel('Eigenvalues')
    ax.errorbar(np.arange(1,nComp+1), np.mean(tmpdata, axis=0), np.std(tmpdata, axis=0), color = 'r', marker='s')    
    ax.set_title(condition)
    x0 = np.arange(1,nComp+1)[np.where((np.mean(tmpdata, axis=0)-np.mean(noiseL, axis=0))>0)[-1][0]]
    
    #ax.set_yticks([0,25,50,75,100])
    #ax12.set_yticks([0,25,50,75,100])
    ax.set_xlabel('# of components')
plt.show()