# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 08:24:57 2018
fig1-S3 - Check covariances from reconstruction.
@author: monika
"""
import numpy as np
import matplotlib as mpl
import os
from scipy.stats import ttest_ind
#
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.ticker as mtick
import dataHandler as dh
# deliberate import all!
from stylesheet import *
################################################
#
# create figure 1: This is twice the normal size
#
################################################
fig = plt.figure('Fig1 - S4 : Covariance from reconstruction.', figsize=(9.5, 9*2/4.))
gs1 = gridspec.GridSpec(12,4)
gs1.update(left=0.055, right=0.99,  bottom = 0.05, top=0.95, hspace=0.1, wspace=0.45)
fig.patch.set_alpha(0.0)


################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML32', 'AML18', 'AML70', 'AML175', 'Special']:
    for condition in ['chip', 'moving', 'immobilized', 'transition']:# ['moving', 'immobilized', 'chip']:
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
transient = ['Special_transition']
gs2 = gridspec.GridSpec(12,5)
gs2.update(left=0.055, right=0.99,  bottom = 0.05, top=0.95, hspace=0.1, wspace=0.45)
for ci, condition, keys in zip([0,1,2,3, 4], [ 'immobilized','moving', 'immobilized (Ctrl)','moving (Ctrl)', 'transition'], [ imExp,movExp,imCtrl, movCtrl, transient ]):
    index = 0
    for key in keys:
        dset = data[key]['analysis']
        
        for idn in dset.keys():
            
            
            #order=  dset[idn]['PCA']['neuronOrderPCA']
            results=  dset[idn]['PCA']['covariance']#[order]
            #results = results[:,order]
            
            ax = plt.subplot(gs2[index, ci])
            ax.imshow(results, aspect='auto', vmin=-1, vmax=1)
            if index==0:
                ax.set_title(condition)
            index+=1
        print index


plt.show()
#################################################
##
## show difference in covariance each half
##
#################################################
# variance explained for moving and immobile 

for ci, condition, keys in zip([0,1,2,3, 4], [ 'immobilized','moving', 'immobilized (Ctrl)','moving (Ctrl)', 'transition'], [ imExp,movExp,imCtrl, movCtrl, transient ]):
    index = 0
    for key in keys:
        dset = data[key]['analysis']
        
        for idn in dset.keys():
            
            
            order=  dset[idn]['PCAHalf1']['neuronOrderPCA']
            results1=  dset[idn]['PCAHalf1']['covariance'][order]
            results2=  dset[idn]['PCAHalf2']['covariance'][order]
            results = results1[:,order]-results2[:,order]
            
            ax = plt.subplot(gs2[index, ci])
            ax.imshow(results, aspect='auto', vmin=-0.5, vmax=0.5)
            if index==0:
                ax.set_title(condition)
            index+=1
        print index
plt.show()


