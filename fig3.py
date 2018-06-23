
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

from stylesheet import *
################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML32', 'AML18', 'AML70', 'AML175']:
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

fig = plt.figure('Fig - 3 : Predicting behavior from neural dynamics', figsize=(9.5, 9*3/4.))
# this gridspec makes one example plot of a heatmap with its PCA
gs1 = gridspec.GridSpec(3, 4, width_ratios = [1,1,0.5,0.5], height_ratios = [1,1,1])
gs1.update(left=0.1, right=0.98, wspace=0.45, bottom = 0.1, top=0.97, hspace=0.3)

# add a,b,c letters, 9 pt final size = 18pt in this case
#letters = ['A', 'B', 'C']
#y0 = 0.99
#locations = [(0,y0),  (0.47,y0), (0.76,y0)]
#for letter, loc in zip(letters, locations):
#    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
#            horizontalalignment='left',verticalalignment='top',)
################################################
#
#first row - LASSO Schematic
#
################################################
# weights
flag = 'ElasticNet'
avWeights = movingAnalysis[flag]['AngleVelocity']['weights']
avRelevant = np.where(np.abs(avWeights>0))[0]
tWeights = movingAnalysis[flag]['Eigenworm3']['weights']
tRelevant = np.where(np.abs(tWeights>0))[0]
notRelevant = (np.where(np.abs(avWeights==0)*np.abs(tWeights==0)))[0]

# one example
time = moving['Neurons']['TimeFull']
#avNeurons = moving['Neurons']['ActivityFull'][np.where(np.abs(avWeights))>0]
#tNeurons = moving['Neurons']['ActivityFull'][np.where(np.abs(tWeights))>0]
label = 'AngleVelocity'
splits = movingAnalysis['Training']
train, test = splits[label]['Train'], splits[label]['Test']
gsScheme = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs1[0,0:2],\
width_ratios=[1.2,1], wspace=0.4, hspace=0.1)

# schematic of behavior prediction Elasitic net
axscheme1 = plt.subplot(gsScheme[0,0])
axscheme1.axis('equal')
axscheme2 = plt.subplot(gsScheme[0,1], sharey=axscheme1)
# input are PCs, let's only show the testset
#
indices = np.concatenate([tRelevant[0:1], avRelevant[:1], notRelevant[0:1]])
lines = moving['Neurons']['Activity'][:, test]
lines = lines[indices]
t = moving['Neurons']['Time'][test]
scale = np.ptp(t)*0.8
ylocs = np.linspace(0,scale*0.8,len(lines))
# for weight circles
ycircle1 = ylocs +scale/12.
ycircle2 = ylocs -scale/12.
xcirc = t[-1]+scale/7
for lindex, line in enumerate(lines):
    line -=np.mean(line)
    line /=np.max(line)
    line*=scale/7.
    color1 = color2 = N1
    if indices[lindex] in avRelevant:
        color1 = R1
    if indices[lindex] in tRelevant:
        color2 = B1
    axscheme1.plot(t,line+ylocs[lindex], color='k')
    axscheme1.text(t[0], ylocs[lindex]+scale/7, 'Neuron {}'.format(indices[lindex]+1), horizontalalignment='center')
    # circles for weights - red
    circ = mpl.patches.Circle((xcirc, ycircle1[lindex]), scale/10.,fill=True,color='w',lw=2, ec=color1, clip_on=False)
    axscheme1.text(xcirc, ycircle1[lindex], r'$w_{}$'.format(lindex+1),color=color1, verticalalignment='center', horizontalalignment='center')
    axscheme1.add_patch(circ)
    #blue circles
    circ = mpl.patches.Circle((xcirc, ycircle2[lindex]), scale/10.,fill=True,linestyle=(0, (1, 1)),color='w',lw=2, ec=color2, clip_on=False)
    axscheme1.text(xcirc, ycircle2[lindex], r'$w_{}$'.format(lindex+1),color=color2, verticalalignment='center', horizontalalignment='center')
    axscheme1.add_patch(circ)
    
ybeh = [ylocs[-1], ylocs[0]+scale/7.]
for behavior, color, cpred, yl, label in zip(['AngleVelocity','Eigenworm3' ], \
            [N1, N1], [R1, B1], ybeh, ['Wave speed', 'Turn']):
    beh = moving['Behavior'][behavior][test]
    meanb, maxb = np.mean(beh),np.std(beh)
    beh = (beh-meanb)/maxb
    beh*=scale/10
    behPred = movingAnalysis[flag][behavior]['output'][test]
    behPred = (behPred-meanb)/maxb
    behPred*=scale/10
    axscheme2.plot(t, beh+yl, color=color)
    axscheme2.plot(t, behPred+yl, color=cpred)
    axscheme2.text(t[-1], yl+scale/5, \
    r'$R^2 = {:.2f}$'.format(np.float(movingAnalysis[flag][behavior]['scorepredicted'])), horizontalalignment = 'right')
    axscheme2.text(t[-1]*1.1, yl, label, rotation=90, color=cpred, verticalalignment='center')
#axscheme2.set_zorder(-1)

axscheme2.set_facecolor('none')
for i in range(len(lines)):
    if indices[i] in avRelevant:
        con = mpl.patches.ConnectionPatch(xyA=(xcirc,ycircle1[i]), xyB=(t[0],ybeh[0]), coordsA="data", coordsB="data",
                              axesA=axscheme1, axesB=axscheme2, color=R1)
        axscheme1.add_artist(con)
        con.set_zorder(-10)    
    if indices[i] in tRelevant:
        con = mpl.patches.ConnectionPatch(xyA=(xcirc,ycircle2[i]), xyB=(t[0], ybeh[1]), coordsA="data", coordsB="data",
                              axesA=axscheme1, axesB=axscheme2, color=B1, lw=2, linestyle=':')
        axscheme1.add_artist(con)
        con.set_zorder(-10)
# add scalebar
l =120
y = ylocs[0] - scale/4.
axscheme1.plot([t[0], t[0]+l],[y, y], 'k', lw=2)
axscheme1.text(t[0]+l*0.5,y*0.8, '2 min', horizontalalignment='center')
axscheme2.plot([t[0], t[0]+l],[y, y], 'k', lw=2)
axscheme2.text(t[0]+l*0.5,y*0.8, '2 min', horizontalalignment='center')

for axtmp in [axscheme1, axscheme2]:
    axtmp.spines['left'].set_visible(False)
    axtmp.spines['bottom'].set_visible(False)
    axtmp.set_yticks([])
    axtmp.set_xticks([])
    #axtmp.set_xlabel('Time (s)')

axscheme1.set_xlim([t[0], 500])
#
#move left
moveAxes(axscheme1, 'left', 0.05)
moveAxes(axscheme2, 'left', 0.05)
moveAxes(axscheme2, 'down', 0.02)
moveAxes(axscheme1, 'scale', 0.02)
moveAxes(axscheme2, 'scale', 0.02)

# plot number of neurons - simpler plot
gsLasso2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs1[0,2:], width_ratios=[1,1], wspace=0.3, hspace=0.1)

axNav= plt.subplot(gsLasso2[0,0])
axNt = plt.subplot(gsLasso2[0,1], zorder=-10)

# number of neurons
for behavior, colors, axR2 in zip(['AngleVelocity', 'Eigenworm3'], [(R2, N0), (B2, N0)], [axNav, axNt ]):

    alldata = []
    # experiment
    c = colors[0]
    for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
        dset = data[key]['analysis']
        keep = []
        for idn in dset.keys():
            results=  dset[idn][flag][behavior]
            keep.append(results['noNeurons'])
            
        keep = np.array(keep)
        rnd1 = np.random.rand(len(keep))*0.2
        axR2.scatter(np.zeros(len(keep))+rnd1, keep, marker = marker, c = c, edgecolor=c, alpha=0.5)
        alldata.append(keep)
    alldata = np.array(alldata)
    mp.mkStyledBoxplot(axR2, [-0.5, 1.5], alldata.T, [c], ['GCamp6s'], scatter=False)
    # controls
    c = colors[1]
    ctrldata = []
    xoff = 1.5
    for key, marker in zip(['AML18_moving', 'AML175_moving'],['o', "p"]):
        dset = data[key]['analysis']
        keep = []
        for idn in dset.keys():
            results=  dset[idn][flag][behavior]
            keep.append(results['noNeurons'])
        keep = np.array(keep)
        rnd1 = np.random.rand(len(keep))*0.2
        axR2.scatter(xoff+np.zeros(len(keep))+rnd1, keep, marker = marker,c = c, edgecolor=c, alpha=0.5)
        ctrldata.append(keep)
    ctrldata = np.array(ctrldata)
    mp.mkStyledBoxplot(axR2, [-0.5+xoff, 1.5+xoff], ctrldata.T, [c,], ['Control (GFP)'], scatter=False)
    
    axR2.set_xlim([-1, 2.5])
    axR2.set_xticks([-0.5,-0.5+xoff])
    axR2.set_xticklabels(['GCaMP6s', 'GFP'])
axNav.set_ylabel('Relevant neurons')
axNav.set_ylim([0,45])
axNt.set_ylim([0,45])

################################################
#
#second row
#
################################################
gsLasso = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs1[1,:2], width_ratios=[1,1],height_ratios=[5,1], wspace=0.2, hspace=0.1)

# Plot test results!

axR2AVa = plt.subplot(gsLasso[0,0])
axR2Ta = plt.subplot(gsLasso[:,1])
# for broken axis
axR2AVb = plt.subplot(gsLasso[1,0])#, sharex = axR2AVa)
#axR2Tb = plt.subplot(gsLasso[1,1])#, sharex = axR2Ta)
for axR2AV, axR2T in [[axR2AVa, axR2Ta], [axR2AVb, axR2Ta]]:
    for behavior, colors, axR2 in zip(['AngleVelocity', 'Eigenworm3'], [(R2, N0), (B2, N0)], [axR2AV, axR2T ]):
    
        alldata = []
        
        # experiment
        c = colors[0]
        for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
            dset = data[key]['analysis']
            keep = []
            for idn in dset.keys():
                results=  dset[idn][flag][behavior]
                try:
                    keep.append(np.array([results['scorepredicted'], np.max(results['individualScore'])]))
                except ValueError:
                     keep.append(np.array([results['scorepredicted'], 0]))
            keep = np.array(keep)
            rnd1 = np.random.rand(len(keep))*0.2
            rnd2 = np.random.rand(len(keep))*0.2
            axR2.scatter(np.zeros(len(keep))+rnd1, keep[:,0], marker = marker, c = c, edgecolor=c, alpha=0.5)
            axR2.scatter(np.ones(len(keep))+rnd2, keep[:,1], marker = marker, c = c, edgecolor=c,alpha=0.7)
            axR2.plot(np.vstack([rnd1, 1+rnd2]), keep.T, color=c,zorder=-5, linestyle=':')
            alldata.append(keep)
        alldata = np.concatenate(alldata)  
        mp.mkStyledBoxplot(axR2, [-0.5,1.5], alldata.T, [c, c], ['Groups of neurons', 'Single neuron'], scatter=False)
        # controls
        c = colors[1]
        ctrldata = []
        xoff = 3
        for key, marker in zip(['AML18_moving', 'AML175_moving'],['o', "p"]):
            dset = data[key]['analysis']
            keep = []
            for idn in dset.keys():
                results=  dset[idn][flag][behavior]
                try:
                    keep.append(np.array([results['scorepredicted'], np.max(results['individualScore'])]))
                except ValueError:
                     keep.append(np.array([results['scorepredicted'], 0]))
            keep = np.array(keep)
            rnd1 = np.random.rand(len(keep))*0.2
            rnd2 = np.random.rand(len(keep))*0.2
            axR2.scatter(xoff+np.zeros(len(keep))+rnd1, keep[:,0], marker = marker,c = c, edgecolor=c, alpha=0.5)
            axR2.scatter(xoff+np.ones(len(keep))+rnd2, keep[:,1], marker = marker, c = c, edgecolor=c, alpha=0.5)
            axR2.plot(np.vstack([rnd1, 1+rnd2])+xoff, keep.T, color=N2, zorder=-2, linestyle=':')
            ctrldata.append(keep)
        ctrldata = np.concatenate(ctrldata)
        
        mp.mkStyledBoxplot(axR2, [-0.5+xoff,1.5+xoff], ctrldata.T, [c, c], ['Groups of neurons', 'Single neuron'], scatter=False)
        
        axR2.set_xlim([-1, 5])
        #axR2.set_xticks([-0.5,1.5, -0.5+xoff,1.5+xoff])
        #axR2.set_xticklabels(['All', 'Single','All', 'Single'])
        axR2.set_xticks([0.5, 0.5+xoff])
        axR2.set_xticklabels(['GCaMP6s', 'GFP'], rotation = 0)
        
axR2AVa.set_ylim([-0.45,0.65])
axR2AVb.set_ylim([-3,-1.2])
axR2Ta.set_ylim([-0.25,0.65])
#axR2Tb.set_ylim([-3,-1.2])
axR2AVa.text(-0.35,-0.1, r'$R^2$ (Testset)', fontsize =mpl.rcParams["axes.labelsize"], verticalalignment ='bottom', rotation=90,transform=axR2AVa.transAxes)
# remove labels and spines
axR2AVa.spines['bottom'].set_visible(False)
#axR2Ta.spines['bottom'].set_visible(False)
axR2AVa.set_xticks([])
#axR2Ta.set_xticks([])
# add fancy linebreaks
d = .015 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=axR2AVa.transAxes, color='k', clip_on=False)
axR2AVa.plot((-d,d), (-d,+d), **kwargs)
kwargs = dict(transform=axR2AVb.transAxes, color='k', clip_on=False)
axR2AVb.plot((-d,d),(1-d*2,1+d*2), **kwargs)

#kwargs = dict(transform=axR2Ta.transAxes, color='k', clip_on=False)
#axR2Ta.plot((-d,d), (-d,+d), **kwargs)
#kwargs = dict(transform=axR2Tb.transAxes, color='k', clip_on=False)
#axR2Tb.plot((-d,d),(1-d*2,1+d*2), **kwargs)
##

#
################################################
#
#  converse prediction - turns from velocity
#
################################################
gsLasso = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs1[2,:2], width_ratios=[1,1],height_ratios=[3,1], wspace=0.2, hspace=0.1)

# Plot test results!
flag = 'ConversePredictionEN'
axR2AVa = plt.subplot(gsLasso[0,0], zorder=-10, fc='none')
axR2Ta = plt.subplot(gsLasso[0,1], zorder=-10, fc='none')
# for broken axis
axR2AVb = plt.subplot(gsLasso[1,0], zorder=-10, fc='none')#, sharex = axR2AVa)
axR2Tb = plt.subplot(gsLasso[1,1], zorder=-10, fc='none')#, sharex = axR2Ta)
for axR2AV, axR2T in [[axR2AVa, axR2Ta], [axR2AVb, axR2Tb]]:
    for behavior, colors, axR2 in zip(['AngleVelocity', 'Eigenworm3'], [(R2, N0), (B2, N0)], [axR2AV, axR2T ]):
    
        alldata = []
        
        # experiment
        c = colors[0]
        for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
            dset = data[key]['analysis']
            keep = []
            for idn in dset.keys():
                results=  dset[idn][flag][behavior]
                
                keep.append(results['scorepredicted'])
                
            keep = np.array(keep)
            rnd1 = np.random.rand(len(keep))*0.2
            rnd2 = np.random.rand(len(keep))*0.2
            axR2.scatter(np.zeros(len(keep))+rnd1, keep, marker = marker, c = c, edgecolor=c, alpha=0.5)
            #axR2.scatter(np.ones(len(keep))+rnd2, keep[:,1], marker = marker, c = c, edgecolor=c,alpha=0.7)
            #axR2.plot(np.vstack([rnd1, 1+rnd2]), keep.T, color=c,zorder=-5, linestyle=':')
            alldata.append(keep)
        alldata = np.array(alldata)  
        mp.mkStyledBoxplot(axR2, [-0.5,1.5], alldata.T, [c], ['Groups of neurons', 'Single neuron'], scatter=False)
        # controls
        c = colors[1]
        ctrldata = []
        xoff = 3
        for key, marker in zip(['AML18_moving', 'AML175_moving'],['o', "p"]):
            dset = data[key]['analysis']
            keep = []
            for idn in dset.keys():
                results=  dset[idn][flag][behavior]
                
                keep.append(results['scorepredicted'])
            keep = np.array(keep)
            rnd1 = np.random.rand(len(keep))
            rnd2 = np.random.rand(len(keep))*0.5
            axR2.scatter(xoff+np.zeros(len(keep))+rnd1, keep, marker = marker,c = c, edgecolor=c, alpha=0.5)
            #axR2.scatter(xoff+np.ones(len(keep))+rnd2, keep[:,1], marker = marker, c = c, edgecolor=c, alpha=0.5)
            #axR2.plot(np.vstack([rnd1, 1+rnd2])+xoff, keep.T, color=N2, zorder=-2, linestyle=':')
            ctrldata.append(keep)
        ctrldata = np.array(ctrldata)
        
        mp.mkStyledBoxplot(axR2, [-0.5+xoff,1.5+xoff], ctrldata.T, [c], ['Groups of neurons', 'Single neuron'], scatter=False)
        
        axR2.set_xlim([-1, 5])
        #axR2.set_xticks([-0.5,1.5, -0.5+xoff,1.5+xoff])
        #axR2.set_xticklabels(['All', 'Single','All', 'Single'])
        axR2.set_xticks([-0.25, xoff-0.25])
        axR2.set_xticklabels(['GCaMP6s', 'GFP'], rotation = 0)
        
axR2AVa.set_ylim([-0.8,0.65])
axR2AVb.set_ylim([-3,-1])
axR2Ta.set_ylim([-0.25,0.65])
axR2Tb.set_ylim([-0.8,-0.4])
axR2AVa.text(-0.35,-0.1, r'$R^2$ (Testset)', fontsize =mpl.rcParams["axes.labelsize"], verticalalignment ='bottom', rotation=90,transform=axR2AVa.transAxes)
# remove labels and spines
axR2AVa.spines['bottom'].set_visible(False)
axR2Ta.spines['bottom'].set_visible(False)
axR2AVa.set_xticks([])
axR2Ta.set_xticks([])
# add fancy linebreaks
d = .015 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=axR2AVa.transAxes, color='k', clip_on=False)
axR2AVa.plot((-d,d), (-d,+d), **kwargs)
kwargs = dict(transform=axR2AVb.transAxes, color='k', clip_on=False)
axR2AVb.plot((-d,d),(1-d*2,1+d*2), **kwargs)
#
kwargs = dict(transform=axR2Ta.transAxes, color='k', clip_on=False)
axR2Ta.plot((-d,d), (-d,+d), **kwargs)
kwargs = dict(transform=axR2Tb.transAxes, color='k', clip_on=False)
axR2Tb.plot((-d,d),(1-d*2,1+d*2), **kwargs)
#

################################################
#
#  weight locations
#
################################################
weightLocs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs1[1:,2:], wspace=0.2, hspace=0.1)
axweight = plt.subplot(weightLocs[0,0],)
X = moving['Neurons']['Positions']
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
xS, yS = pca.fit_transform(X.T).T

# plot projections of neurons
s0,s1,s2 = 64, 64, 32 # size of gray, red, blue neurons

axweight.scatter(xS,yS,color=N2, s = s0)
##circle_scatter(ax1, xS, yS, radius=s0, color=UCgray[0])
axweight.scatter(xS[avRelevant],yS[avRelevant],color=R2, s = s1)
axweight.scatter(xS[tRelevant],yS[tRelevant],color=B1, s = s2)
#
axatlas= plt.subplot(weightLocs[1,0],)
neuron2D = 'utility/celegans277positionsKaiser.csv'
labels = np.loadtxt(neuron2D, delimiter=',', usecols=(0), dtype=str)
neuronAtlas2D = np.loadtxt(neuron2D, delimiter=',', usecols=(1,2))
relevantIds = (neuronAtlas2D[:,0]>-0.0)#*(Xref[:,0]<0.1)
Xref = neuronAtlas2D[relevantIds]
Xref[:,0] = -Xref[:,0]
labels = labels[relevantIds]
xR, yR = pca.fit_transform(Xref).T
axatlas.scatter(xR,yR,color=N2, s = s0)
plt.show()