
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:15:14 2018
Figure 2 - Behavior is represented in the brain
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



#import singlePanels as sp
#import makePlots as mp
import dataHandler as dh
# deliberate import all!
from stylesheet import *
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

################################################
#
# create figure 1: This is twice the normal size
#
################################################
# we will select a 'special' dataset here, which will have all the individual plots


fig = plt.figure('Fig - S1 : Neural dynamics in immobile transitions', figsize=(9.5, 9*2/4.))
# this gridspec makes one example plot of a heatmap with its PCA
#gs1 = gridspec.GridSpec(4, 3, width_ratios = [1,1,1], height_ratios=[0.1, 1,1,2])
#gsHeatmap = gridspec.GridSpecFromSubplotSpec(4,5, subplot_spec=gs1[0:4,:], width_ratios=[1.25, 0.1, 0.5,0.5,0.5], height_ratios = [0.1,10,10,10], wspace=0.3, hspace=0.25)
gs1 = gridspec.GridSpec(2,4)
gs1.update(left=0.07, right=0.98,  bottom = 0.15, top=0.98, hspace=0.25, wspace=0.55)

################################################
#
# letters
#
################################################

# add a,b,c letters, 9 pt final size = 18pt in this case
#letters = ['A', 'B', 'C']
#y0 = 0.99
#locations = [(0,y0),  (0.55,y0), (0.76,y0)]
#for letter, loc in zip(letters, locations):
#    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
#            horizontalalignment='left',verticalalignment='top',)
#
#letters = ['D', 'E', 'F']
#y0 = 0.6
#locations = [(0,y0),  (0.40,y0), (0.76,y0)]
#for letter, loc in zip(letters, locations):
#    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
#            horizontalalignment='left',verticalalignment='top',)
#
#letters = ['G', 'H', 'I']
#y0 = 0.29
#locations = [(0,y0),  (0.19,y0), (0.40,y0)]
#for letter, loc in zip(letters, locations):
#    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
#            horizontalalignment='left',verticalalignment='top',)

#letters = ['I', 'J']
#y0 = 0.27
#locations = [(0,y0),  (0.22,y0), (0.76,y0)]
#for letter, loc in zip(letters, locations):
#    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
#            horizontalalignment='left',verticalalignment='top',)
################################################
#
#first row -- show example heatmaps for a movng and immobile dataset
#
################################################
fig = plt.figure('Fig - 1 : Neural dynamics in freely moving animals', figsize=(9.5, 9*3/4.))
gsHeatmap = gridspec.GridSpec(3,4,  width_ratios=[1, 1, 1, 0.1], height_ratios = [1,0.1,0.75])
gsHeatmap.update(left=0.075, right=0.98,  bottom = 0.11, top=0.98, hspace=0.3, wspace=0.45)
fig.patch.set_alpha(0.0)
#heatmap axes
axhm1 = plt.subplot(gsHeatmap[0,0])
axhm1 = plt.subplot(gsHeatmap[0,1])
axhm1 = plt.subplot(gsHeatmap[0,2])
axcb = plt.subplot(gsHeatmap[0,3])
# ethogram
axetho = plt.subplot(gsHeatmap[1,0], clip_on=False)
axetho2 = plt.subplot(gsHeatmap[1,1], clip_on=False)
axetho3 = plt.subplot(gsHeatmap[1,2], clip_on=False)
# legend for ethogram
axEthoLeg = plt.subplot(gsHeatmap[1,3])#,clip_on=False)
# principal components
axpc1 =plt.subplot(gsHeatmap[2,0], clip_on=False)#, sharex=axhm)
axpc2 =plt.subplot(gsHeatmap[2,1], clip_on=False)#, sharex=axhm)
axpc3 =plt.subplot(gsHeatmap[2,2], clip_on=False)#, sharex=axhm)
plt.show()
################################################
#
#first row -- show example heatmaps for a movng and immobile dataset
#
################################################
#movData = 'BrainScanner20170613_134800'
#immData = 'BrainScanner20180510_092218'
#transientData = 'BrainScanner20180511_134913'
#for key, dset, label, ax in zip(['AML32_moving', 'AML32_immobilized', 'AML32_chip'],[movData, immData, transientData], ['moving', 'immobilized', 'transient'], [ax8, ax9, ax10]):
#    

plt.show()