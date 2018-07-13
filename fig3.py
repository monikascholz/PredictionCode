
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:15:14 2018
Figure 3 - Sparse linear model predicts behavior
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
from  sklearn.metrics.pairwise import pairwise_distances
from matplotlib_venn import venn2
 
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
# custom pip
#import svgutils as svg
#
import makePlots as mp
import dataHandler as dh
import dimReduction as dr

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
# negative bends are ventral for this worm
fig = plt.figure('Fig - 3 : Predicting behavior from neural dynamics', figsize=(9.5, 6*2.25))
# this gridspec makes one example plot of a heatmap with its PCA
gs1 = gridspec.GridSpec(5, 4, width_ratios = [1,1, 0.5,0.5], height_ratios = [1,1,0.75,0.75, 2])
gs1.update(left=0.1, right=0.98, wspace=0.45, bottom = 0.01, top=0.97, hspace=0.2)

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
gsScheme = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec=gs1[0,0:2],\
width_ratios=[1.2,1, 1],height_ratios=[1,2], wspace=0.2, hspace=0.1)

# schematic of behavior prediction Elasitic net
axscheme1 = plt.subplot(gsScheme[:,0])
axscheme1.axis('equal')
axscheme2 = plt.subplot(gsScheme[:,1], sharey=axscheme1)
axvenn = plt.subplot(gsScheme[0,2],)
axvennBox = plt.subplot(gsScheme[1,2])
# input are PCs, let's only show the testset
#
indices = np.concatenate([tRelevant[0:1], avRelevant[:1], notRelevant[0:1]])
lines = moving['Neurons']['Activity'][:, test]
lines = lines[indices]
t = moving['Neurons']['Time'][test]
scale = np.ptp(t)*0.8
ylocs = np.linspace(0,scale*0.9,len(lines))
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
    axscheme1.text(t[0]*1.1, ylocs[lindex]+scale/7, 'Neuron {}'.format(indices[lindex]+1), horizontalalignment='center')
    # circles for weights - red
    circ = mpl.patches.Circle((xcirc, ycircle1[lindex]), scale/8.,fill=True,color='w',lw=2, ec=color1, clip_on=False)
    axscheme1.text(xcirc, ycircle1[lindex], r'$w_{}$'.format(lindex+1),color=color1, verticalalignment='center', horizontalalignment='center')
    axscheme1.add_patch(circ)
    #blue circles
    circ = mpl.patches.Circle((xcirc, ycircle2[lindex]), scale/8.,fill=True,linestyle=(0, (1, 1)),color='w',lw=2, ec=color2, clip_on=False)
    axscheme1.text(xcirc, ycircle2[lindex], r'$w_{}$'.format(lindex+1),color=color2, verticalalignment='center', horizontalalignment='center')
    axscheme1.add_patch(circ)
    
ybeh = [ylocs[-1], ylocs[0]+scale/7.]
for behavior, color, cpred, yl, label in zip(['AngleVelocity','Eigenworm3' ], \
            [N1, N1], [R1, B1], ybeh, ['WS', 'T']):
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
    # label the behaviors
    axscheme2.text(t[-1]*1.05, yl, label, rotation=0, color=cpred, verticalalignment='center')
#axscheme2.set_zorder(-1)
# circles around weights
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
y = ylocs[0] - scale/3.5
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
moveAxes(axscheme1, 'left', 0.07)
moveAxes(axscheme2, 'left', 0.07)
#moveAxes(axscheme2, 'down', 0.02)
#moveAxes(axscheme1, 'scale', 0.02)
#moveAxes(axscheme2, 'scale', 0.02)

# venn diagram for number of neurons in each group and boxplot
venn = []

for key in ['AML32_moving', 'AML70_chip']:
        dset = data[key]['analysis']
        keep = []
        for idn in dset.keys():
            
            results=  dset[idn][flag]
            avWeights = results['AngleVelocity']['weights']
            avRelevant = np.where(np.abs(avWeights>0))[0]
            tWeights = results['Eigenworm3']['weights']
            tRelevant = np.where(np.abs(tWeights>0))[0]
            notRelevant = (np.where(np.abs(avWeights==0)*np.abs(tWeights==0)))[0]
            N = 1.0*len(avWeights)
            unique = len(np.unique(np.concatenate([avRelevant, tRelevant])))
            overlap = len(np.intersect1d(avRelevant, tRelevant))
            zero = len(notRelevant)
            print unique, overlap, zero
            venn.append([unique/N, overlap/N, zero/N])
            if idn ==movingAML32:
                movingVenn=[len(avRelevant), len(tRelevant), len(np.intersect1d(avRelevant, tRelevant))]
print movingVenn
venndiagram = venn2(subsets=movingVenn, set_labels=['WS', 'T'], set_colors=(R2, B2), ax=axvenn, alpha=0.9)
# move labels
venndiagram.get_label_by_id('A').set_y(0.2)
venndiagram.get_label_by_id('A').set_x(-0.75)
venndiagram.get_label_by_id('B').set_y(0.2)
venndiagram.get_label_by_id('B').set_x(1.0)
venndiagram.get_label_by_id('A').set_fontsize(14)
venndiagram.get_label_by_id('B').set_fontsize(14)
overlapColor = venndiagram.get_patch_by_id('110').get_fc()

moveAxes(axvenn, 'up', 0.025)
moveAxes(axvennBox, 'up', 0.025)
moveAxes(axvenn, 'scale', 0.02)
moveAxes(axvennBox, 'scaley', -0.025)
#moveAxes(axR2, 'up', 0.025)
# plot boxplot of overlap
mp.mkStyledBoxplot(axvennBox, [0,1,2], np.array(venn).T, [L1, overlapColor, N0], ['unique','overlap', 'zero'])
axvennBox.set_xlim([-0.5, 2.25])
axvennBox.set_ylabel('Fraction', labelpad=0)
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
    moveAxes(axR2, 'scaley', -0.025)
    moveAxes(axR2, 'up', 0.025)
axNav.set_ylabel('# neurons')
axNav.set_ylim([0,45])
axNt.set_ylim([0,45])

################################################
#
#second row
#
################################################
gsLasso = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs1[1,:2], width_ratios=[1,1],height_ratios=[5,1], wspace=0.35, hspace=0.15)

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
axR2AVb.plot((-d,d),(1-d*5,1+d*5), **kwargs)

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
gsLasso = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs1[1,2:], width_ratios=[1,1],height_ratios=[3,1], wspace=0.35, hspace=0.1)

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
        xoff = 1.5
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
        
        axR2.set_xlim([-1, 2.5])
        #axR2.set_xticks([-0.5,1.5, -0.5+xoff,1.5+xoff])
        #axR2.set_xticklabels(['All', 'Single','All', 'Single'])
        axR2.set_xticks([-0.25, xoff-0.25])
        axR2.set_xticklabels(['GCaMP6s', 'GFP'], rotation = 30)
        
axR2AVa.set_ylim([-0.8,0.65])
axR2AVb.set_ylim([-3,-1])
axR2Ta.set_ylim([-0.25,0.65])
axR2Tb.set_ylim([-0.8,-0.4])
axR2AVa.text(-0.6,-0.2, r'$R^2$ (Testset)', fontsize =mpl.rcParams["axes.labelsize"], verticalalignment ='bottom', rotation=90,transform=axR2AVa.transAxes)
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
axR2AVb.plot((-d,d),(1-d*3,1+d*2), **kwargs)
#
kwargs = dict(transform=axR2Ta.transAxes, color='k', clip_on=False)
axR2Ta.plot((-d,d), (-d,+d), **kwargs)
kwargs = dict(transform=axR2Tb.transAxes, color='k', clip_on=False)
axR2Tb.plot((-d,d),(1-d*3,1+d*2), **kwargs)
#

################################################
#
#  Third row - show locations of neurons in all setsa
#
################################################
ventral = [1,1,1,1,1,-1]
# plot projections of neurons
s0,s1,s2 = 16, 16, 16 # size of gray, red, blue neurons
dim = False
flag = 'ElasticNet'
weightLocs = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=gs1[2:4,:], wspace=0.0, hspace=0)
axweight = plt.subplot(weightLocs[0,0], aspect='equal', clip_on=False)
axweight2 = plt.subplot(weightLocs[0,1], aspect='equal', clip_on=False)
axweight3 = plt.subplot(weightLocs[1,0], aspect='equal', clip_on=False)
axweight4 = plt.subplot(weightLocs[1, 1], aspect='equal', clip_on=False)
from pycpd import deformable_registration, rigid_registration
index = 0
markers = ['p', '^', '*', 'X', '+', '8', 's']
# use the moving dataset as reference
Xref = moving['Neurons']['Positions'].T
Xref -=np.mean(Xref,axis=0)
# load atlas data
neuron2D = 'utility/celegans277positionsKaiser.csv'
labels = np.loadtxt(neuron2D, delimiter=',', usecols=(0), dtype=str)
neuronAtlas2D = np.loadtxt(neuron2D, delimiter=',', usecols=(1,2))
relevantIds = (neuronAtlas2D[:,0]>0.0)#*(Xref[:,0]<0.1)
A = neuronAtlas2D[relevantIds]
A[:,0] = -A[:,0]
labels = labels[relevantIds]
A -=np.mean(A, axis=0)
A /= np.ptp(A, axis=0)
A*= 1.2*np.ptp(Xref[:,:2], axis=0)
# register atlas to reference dataset
registration = rigid_registration
reg = registration(Xref[:,:2], A)
reg.register(callback=None)
registration = deformable_registration

reg = registration(Xref[:,:2],reg.TY)
def callback(iteration, error, X, Y):
    return 0
reg.register(callback)
A = reg.TY

# save some interesting neurons
interestingNeurons = []
labels_moving = []
for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
        dset = data[key]['input']
        res = data[key]['analysis']
        for idn in dset.keys():
            
            #results=  dset[idn][flag][behavior]
            X = np.copy(dset[idn]['Neurons']['Positions']).T
            X -=np.mean(Xref,axis=0)
            X[:,1] *=ventral[index]
            xS, yS, _ = X.T
            registration = rigid_registration
            
            reg = registration(Xref, X)
            reg.register(callback=None)
            registration = deformable_registration
            
            reg = registration(Xref, reg.TY)
            reg.register(callback=None)
            xS, yS, zS = reg.TY.T
            #print reg.R
            #from sklearn.decomposition import PCA
            #pca = PCA(n_components = 2)
            #xS, yS = pca.fit_transform(X.T).T
            avWeights =res[idn][flag]['AngleVelocity']['weights']
            avRelevant = np.where(np.abs(avWeights>0))[0]
            avRelevant = avRelevant[np.argsort(np.abs(avWeights[avRelevant]))]
            # check if prediction is good otehrwise pass
            if res[idn][flag]['AngleVelocity']['scorepredicted']<0.2:
                avRelevant = []
            tWeights = res[idn][flag]['Eigenworm3']['weights']
            tRelevant = np.where(np.abs(tWeights>0))[0]
            tRelevant = tRelevant[np.argsort(np.abs(tWeights[tRelevant]))]
            # check if prediction is good otehrwise pass
            if res[idn][flag]['Eigenworm3']['scorepredicted']<0.2:
                tRelevant = []
            
            if dim:
                axweight = plt.subplot(weightLocs[ index],projection='3d')
                axweight.scatter(xS,yS,zS,color=N1, s = s0, facecolor='None')
                axweight.scatter(xS[avRelevant],yS[avRelevant],zS[avRelevant],color=R2, s = s1, marker='^')
                axweight.scatter(xS[tRelevant],yS[tRelevant],zS[tRelevant],color=B1, s = s2, marker='*')
            else:
                
                if len(avRelevant)>0:
                    axweight.scatter(xS,yS,color=N1, s = s0, alpha=0.2)
                    axweight.scatter(xS[avRelevant],yS[avRelevant],color=R2, s = np.linspace(8,64,len(avRelevant)), marker=markers[index], alpha=0.5)
                    
                
                axweight2.scatter(xS,yS,color=N1, s = s0, alpha=0.2)
                axweight2.scatter(xS[tRelevant],yS[tRelevant],color=B1, s = np.linspace(8,64,len(tRelevant)), marker=markers[index], alpha=0.5)
                
                n = 3
                
                #axweight3.scatter(xS,yS,color=N1, s = s0, alpha=0.2)
                if len(avRelevant)>0:
                    axweight3.scatter(xS[avRelevant[-n:]],yS[avRelevant[-n:]],color=R2, s = s1, marker=markers[index], alpha=0.5)
                
                
                #axweight4.scatter(xS,yS,color=N1, s = s0, alpha=0.2)
                axweight4.scatter(xS[tRelevant[-n:]],yS[tRelevant[-n:]],color=B1, s = s2, marker=markers[index], alpha=0.5)
                # show a few text labels next to neurons
                # calculate distance of neuron to text
                stored = []
                for axes, relevant, c in zip([[axweight, axweight3], [axweight2, axweight4]], [avRelevant, tRelevant], [R1, B1]):
                    for ax in axes:
                        try:
                            
                            D = pairwise_distances(np.vstack([xS[relevant], yS[relevant]]).T, A)
                            # find minimal distances - this is the atlas ID
                            candNeurons = np.argmin(D, axis=1)
                            
                            if idn==movingAML32:
                                labels_moving.append(labels[candNeurons])
                            # find which neuron we matched with for the largest weighted ones
                            
                            
                            for nneur in range(len(candNeurons))[-n:]:
                                if labels[candNeurons[nneur]][:3] in ['AVA', 'ASI', 'AIY','AIB', 'RIM', 'SMB', 'RMD', 'SMD', 'RIV']:
                                    valid = dset[idn]['Neurons']['valid']
                                    # which neuron it is in dataset
                                    rowNeur = relevant[nneur]
                                    if rowNeur not in stored:
                                        interestingNeurons.append([labels[candNeurons[nneur]], dset[idn]['Neurons']['RawActivity'][rowNeur, valid], dset[idn]['Behavior']['AngleVelocity'], dset[idn]['Behavior']['Eigenworm3']])
                                        stored.append(rowNeur)
                                #ax.text(xS[relevant[ nneur]], yS[relevant[ nneur]], labels[candNeurons[nneur]][:4], fontsize=8, alpha=1, color=c, horizontalalignment='center')
                        except ValueError:
                            pass
                
            index +=1
for ax in [axweight, axweight2, axweight3, axweight4]:
    ax.set_xlim([-150, 200])
    ax.set_ylim([-50, 30])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    #moveAxes(ax, 'scale', 0.05)
    #moveAxes(ax, 'down', 0.01)
moveAxes(axweight2, 'down', 0.02)
moveAxes(axweight, 'down', 0.02)
# show atlas in lower plots
axweight4.scatter(A[:,0],A[:,1],color=N0, s = s2, alpha=0.25, zorder=-100)
axweight3.scatter(A[:,0],A[:,1],color=N0, s = s2, alpha=0.25, zorder=-100)
# add orientation bars
length=20
xmin, ymin = axweight.get_xlim()[0], axweight.get_ylim()[0]+length/4.
xmax, ymax = xmin+length, ymin+length

xmid, ymid = np.mean([xmin, xmax]), np.mean([ymin, ymax])
axweight.plot([xmid, xmid], [ymin, ymax], 'k')
axweight.plot([xmin, xmax], [ymid, ymid], 'k')
# move slightly

axweight.text(xmid, ymin, 'D', horizontalalignment = 'center', verticalalignment ='top')
axweight.text(xmid, ymax, 'V', horizontalalignment = 'center', verticalalignment ='bottom')
axweight.text(xmin, ymid, 'A', horizontalalignment = 'right', verticalalignment ='center')
axweight.text(xmax, ymid, 'P', horizontalalignment = 'left', verticalalignment ='center')

################################################
#
#  Fourth row - pull out how mucyh information is in different neurons by hierarchical clustering
#
################################################
print labels_moving
gsDendro = gridspec.GridSpecFromSubplotSpec(4,3, gs1[4, :], width_ratios=[0.5,1,0.35], hspace=0.1, wspace=0.1)
ax1 = plt.subplot(gsDendro[:3,0])
ax2 = plt.subplot(gsDendro[:2,1])
ax3 = plt.subplot(gsDendro[:2,2], zorder=1)
ax4 = plt.subplot(gsDendro[3:,0])
ax5 = plt.subplot(gsDendro[2:,1])
ax6 = plt.subplot(gsDendro[2:,2], sharey=ax3)
axs = [[ax1, ax2, ax3], [ax4, ax5, ax6]]
links = [L1, L2, L3, L0]
set_link_color_palette(links)

neurons = moving['Neurons']['RawActivity']
t = moving['Neurons']['Time']
for b, (behavior, c, lbl) in enumerate(zip(['AngleVelocity', 'Eigenworm3'], [R1, B1], ['Wave speed', 'Turn'])):
    beh =moving['Behavior'][behavior]
    
    Weights =movingAnalysis[flag][behavior]['weights']
    Relevant = np.where(np.abs(Weights>0))[0]
    labels = labels_moving[b]
    for li, lab in enumerate(labels):
        if lab[:3] in ['AVA', 'ASI', 'AIY','AIB', 'RIM', 'SMB', 'RMD', 'SMD', 'RIV']:
            labels[li] = lab
        else:
            labels[li] = ''
    if len(Relevant)<1:
        print 'upps'
        continue
    
    
    pars = None
    subset = Relevant
    clust = dr.runHierarchicalClustering(moving, pars, subset)
    
    dn = dendrogram(clust['linkage'],ax = axs[b][0],leaf_font_size=10, leaf_rotation=0,labels=labels,\
         orientation = 'left', show_leaf_counts=1, above_threshold_color='k', color_threshold= clust['threshold'])
    print dn['ivl'], dn['leaves'], clust['leafs']
    xlbls = axs[b][0].get_ymajorticklabels()
    
    for lbi,lb in enumerate(xlbls):
        lb.set_color(links[clust['leafs'][dn['leaves'][lbi]]-1])

    traces = clust['clusters']
    for n in range( clust['nclusters']):
        print n
        axs[b][1].plot(t, traces[n].T+5*n, 'k', alpha=0.2)
        axs[b][1].plot(t, np.nanmean(traces[n], axis=0)+n*5, color= links[n])
        # sort by behavior and plot
        
        xPlot, avg, std = sortnbin(beh, np.nanmean(traces[n], axis=0), nBins=10, rng=(np.min(beh), np.max(beh)))
        axs[b][2].plot(xPlot, avg+n, color= links[n])
        axs[b][2].fill_between(xPlot, avg-std+n, avg+std+n,color= links[n], alpha=0.5)
        axs[b][2].plot(xPlot,n*np.ones(len(xPlot)), color= 'k', linestyle=':')
        #dashed line at zero
        #axs[b][2].axvline(0,0, 0.8, color= 'k', linestyle=':')
        if b==0:
            axs[b][2].axvspan(-0.002, 0.002, zorder=-10, alpha=0.1, color='k')
        if b==1:
            axs[b][2].axvspan(-3, 3, zorder=-10, alpha=0.1, color='k')
            
        
    
    if b==0:
        axs[0][2].plot([xPlot[0], xPlot[0]], [3,3.5], color=c, lw = 2)
        axs[0][2].text(xPlot[0]+0.005,3.25 , r'$\Delta R/ R_0 = 0.5$', verticalalignment = 'center')
        
    axs[1][2].set_xticks([-10, 10])
    axs[1][2].set_xticklabels(['Ventral',' Dorsal'])
    axs[0][2].set_xticks([-0.02, 0.04])
    axs[0][2].set_xticklabels(['Reverse',' Forward'])
        
    axs[b][2].spines['left'].set_visible(False)
    #axs[b][2].set_xlabel(lbl)
    axs[b][2].set_yticks([])
    axs[0][1].set_xticks([])
    axs[b][0].spines['left'].set_visible(False)
    axs[b][0].set_ylabel(lbl)
    axs[b][1].spines['left'].set_visible(False)
    axs[0][1].spines['bottom'].set_visible(False)
   
    axs[b][1].set_yticks([])
    axs[1][0].spines['bottom'].set_visible(False)
    axs[0][0].spines['bottom'].set_visible(False)
    axs[0][0].set_xticks([])
    axs[1][1].set_xlabel('Time (s)')
    
for ax in [ax1, ax4]:
    moveAxes(ax, 'left', 0.05)
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    #moveAxes(ax, 'scale', 0.025)
    moveAxes(ax, 'up', 0.05)
plt.show()


#ax = plt.subplot(gs1[4,0])
#for ind, y in enumerate(interestingNeurons):
#    #y = interestingNeurons[neur] 
#    ax.plot(ind+y[1][:1500], 'k', lw = 1)
#    ax.plot(ind+y[2][:1500]*10, 'k', lw = 1)
#    #plotEthogram(ax, np.arange(len(y[0]))[:100], y[1][:100], alpha = 1, yValMax=ind+1, yValMin=ind-1, legend=0)
#    #plt.show()
#    ax.text(-50, ind, y[0])
#ax.set_ylim(-4, len(interestingNeurons))


# plot activity as a function of velocity and turns
bins = np.arange(-0.05, 0.06, 0.005)
ybins = np.arange(-0.5, 1.5, 0.01)
gs = gridspec.GridSpecFromSubplotSpec(1,8, gs1[4, :])
for ind, y in enumerate(interestingNeurons[:8]):
    ax = plt.subplot(gs[ind])
    
    H, xe, ye = np.histogram2d(y[2], y[1], (bins, ybins))
    
    m, s = np.nanmean(H*(ye[:-1]+np.diff(ye)[0]*0.5), axis=1), np.nanstd(H, axis=1)/np.sqrt(np.nansum(H, axis=1))
    ax.plot(xe[:-1]+np.diff(xe)[0]*0.5, m, label = y[0])
    ax.fill_between(xe[:-1]+np.diff(xe)[0]*0.5, m-s,m+s, alpha=0.5)
    ax.legend()
#    ax.scatter(np.sort(y[1]), y[0][np.argsort(y[1])])


plt.show()


## predict immobile behavior
#
## load immobile worm from fig1
## select a special dataset - transiently immobilized
#transient = data['AML32_chip']['input']['BrainScanner20180511_134913']
#transientAnalysis = data['AML32_chip']['analysis']['BrainScanner20180511_134913']
## time first half, second half. Indices of times
#timeHalf = np.arange(0, 1400)
#time2Half = np.arange(1600, transient['Neurons']['Activity'].shape[1])
## pull out repeated stuff
#time = transient['Neurons']['TimeFull']
#timeActual = transient['Neurons']['Time']
#noNeurons = transient['Neurons']['Activity'].shape[0]
##
#label = 'AngleVelocity'
#splits = transientAnalysis['Training']
#train, test = splits[label]['Train'], splits[label]['Test']
#t = time[test]
#axImm = plt.subplot(gs1[3, 0])
#
#ethoImm = dh.makeEthogram(transientAnalysis[flag]['AngleVelocity']['output'][test], transientAnalysis[flag]['Eigenworm3']['output'][test])
## plot predicted behavior
#for behavior, color, cpred, yl, label in zip(['AngleVelocity','Eigenworm3' ], \
#            [N1, N1], [R1, B1],[0, 1], ['Wave speed', 'Turn']):
#    #beh = transient['Behavior'][behavior][test]
#    #meanb, maxb = np.mean(beh),np.std(beh)
#    #beh = (beh-meanb)/maxb
#    #beh*=scale/10
#    behPred = transientAnalysis[flag][behavior]['output'][test]
#    #behPred = (behPred-np.mean(behPred))#/np.max(behPred)
#    print transientAnalysis[flag][behavior]['score']
#    print np.max(behPred)
#    #behPred*=scale/10
#    #axImm.plot(t, beh+yl, color=color)
#    #axImm.plot(t, behPred+yl, color=cpred)
#    axImm.text(t[-1], yl+scale/5, \
#    r'$R^2 = {:.2f}$'.format(np.float(transientAnalysis[flag][behavior]['scorepredicted'])), horizontalalignment = 'right')
#    axImm.text(t[-1]*1.1, yl, label, rotation=90, color=cpred, verticalalignment='center')
#    plotEthogram(axImm, t, ethoImm, alpha = 0.5, yValMax=1, yValMin=0, legend=0)