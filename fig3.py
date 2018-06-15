
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
# custom pip
import svgutils as svg
#
import singlePanels as sp
import makePlots as mp
import dataHandler as dh

################################################
#
# define colors
#
################################################
axescolor = 'k'
mpl.rcParams["axes.edgecolor"]=axescolor
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
# text
mpl.rcParams["text.color"]='k'
mpl.rcParams["ytick.color"]=axescolor
mpl.rcParams["xtick.color"]=axescolor
mpl.rcParams["axes.labelcolor"]='k'
mpl.rcParams["savefig.format"] ='pdf'
# change legend properties
mpl.rcParams["legend.frameon"]=False
mpl.rcParams["legend.labelspacing"]=0.25
mpl.rcParams["legend.labelspacing"]=0.25
#mpl.rcParams['text.usetex'] =True
mpl.rcParams["axes.labelsize"]=  12
mpl.rcParams["xtick.labelsize"]=  12
mpl.rcParams["ytick.labelsize"]=  12
mpl.rcParams["axes.labelpad"] = 0
mpl.rc('font', **{'sans-serif' : 'FiraSans','family' : 'sans-serif'})
mpl.rc('text.latex', preamble='\usepackage{sfmath}')
plt.rcParams['image.cmap'] = 'viridis'
################################################
#
# define colors
#
################################################
# shades of red, dark to light
R0, R1, R2 = '#651119ff', '#b0202eff', '#d15144ff'
Rs = [R0, R1, R2]
# shades of blue
B0, B1, B2 = '#2e2f48ff', '#2b497aff', '#647a9eff'
Bs = [B0, B1, B2]
# shades of viridis
V0, V1, V2, V3, V4 = '#403f85ff', '#006e90ff', '#03cea4ff', '#c3de24ff', '#f1e524ff'
Vs = [V0, V1, V2, V3, V4]
# line plot shades
L0, L1, L2, L3 = ['#1a5477ff', '#0d8d9bff', '#ce5c00ff', '#f0a202ff']
Ls = [L0, L1, L2, L3]
# neutrals
N0, N1, N2 = '#383936ff', '#8b8b8bff', '#d1d1d1ff'
Ns = [N0, N1, N2]
# make a transition cmap
transientcmap = mpl.colors.ListedColormap([mpl.colors.to_rgb(B1), mpl.colors.to_rgb(R1)], name='transient', N=None)

################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML32']:
    for condition in ['moving']:# ['moving', 'immobilized', 'chip']:
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

fig = plt.figure('Fig - 3 : Predicting behavior from neural dynamics', figsize=(9.5, 6.75))
# this gridspec makes one example plot of a heatmap with its PCA
gs1 = gridspec.GridSpec(1, 3, width_ratios = [1,0.5,0.5])
gs1.update(left=0.07, right=0.98, wspace=0.45, bottom = 0.1, top=0.97, hspace=0.65)

# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B', 'C']
y0 = 0.99
locations = [(0,y0),  (0.47,y0), (0.76,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
            horizontalalignment='left',verticalalignment='top',)
################################################
#
#first row - LASSO Schematic
#
################################################
# weights
flag = 'ElasticNet'
avWeights = movingAnalysis[flag]['AngleVelocity']['weights']
print len(np.abs(avWeights)>0)
tWeights = movingAnalysis[flag]['Eigenworm3']['weights']
# relevant neurons
time = moving['Neurons']['TimeFull']
avNeurons = moving['Neurons']['ActivityFull'][np.where(np.abs(avWeights))>0]
tNeurons = moving['Neurons']['ActivityFull'][np.where(np.abs(tWeights))>0]
print moving['Neurons']['ActivityFull'].shape
gsLasso = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs1[0,:], width_ratios=[1,1,2], wspace=0.2, hspace=0.35)

# plot neural line data
axAV = plt.subplot(gsLasso[0,0])
axT = plt.subplot(gsLasso[1,0])

axAV.plot(time, avNeurons.T[:,:,0])
axT.plot(time, avNeurons.T[:,:,0])

################################################
#
#second row
#
################################################
            
            
plt.show()