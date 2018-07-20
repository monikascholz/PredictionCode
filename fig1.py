# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:15:14 2018
Figure 1 - Moving animals show more complex dynamics than immobilized ones.
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
#
#import singlePanels as sp
import makePlots as mp
import dataHandler as dh

from stylesheet import *
################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML32', 'AML18']:
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


fig = plt.figure('Fig - 1 : Neural dynamics in freely moving animals', figsize=(9.5,6.75))
# this gridspec makes one example plot of a heatmap with its PCA
gs1 = gridspec.GridSpec(3, 4, width_ratios = [1.2,0.5,1, 0.6], height_ratios=[1,1,1])
gs1.update(left=0.07, right=0.98, wspace=0.45, bottom = 0.1, top=0.97, hspace=0.5)

################################################
#
# first row
#
################################################
ax2 = plt.subplot(gs1[0,1])
ax3 = plt.subplot(gs1[0,2])
ax4 = plt.subplot(gs1[0,3])
# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B', 'C', 'D']
y0 = 0.99
locations = [(0,y0), (0.35,y0), (0.47,y0), (0.76,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
            horizontalalignment='left',verticalalignment='top',)
# select a special dataset - transiently immobilized
transient = data['AML32_chip']['input']['BrainScanner20180511_134913']
transientR = data['AML32_chip']['analysis']['BrainScanner20180511_134913']
# time first half, second half. Indices of times
timeHalf = np.arange(0, 1400)
time2Half = np.arange(1600, transient['Neurons']['Activity'].shape[1])
# pull out repeated stuff
time = transient['Neurons']['Time']
noNeurons = transient['Neurons']['Activity'].shape[0]
results = transientR['PCA']
resultshalf = transientR['PCAHalf1']
results2half = transientR['PCAHalf2']
# plot heatmap ordered by PCA
# colorbar in a nested gridspec because its much better          
gsHeatmap = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs1[0,0], width_ratios=[10,1], height_ratios = [1.5,10], wspace=0.1, hspace=0.2)
axhm = plt.subplot(gsHeatmap[1,0])
axcb = plt.subplot(gsHeatmap[1,1])
axetho = plt.subplot(gsHeatmap[0,0])
plotEthogram(axetho, time, transient['Behavior']['Ethogram'], alpha = 1, yValMax=1, yValMin=0, legend=0)
axetho.set_xticks([])
axetho.xaxis.label.set_visible(False)
#axetho.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
cax1 = plotHeatmap(transient['Neurons']['TimeFull'], transient['Neurons']['RawActivity'][results['neuronOrderPCA']], ax=axhm, vmin=-1, vmax=1.5)

axhm.set_xlabel('Time (s)')
cbar = fig.colorbar(cax1, cax=axcb, use_gridspec = True)
cbar.set_ticks([-2,0,2])
cbar.set_ticklabels(['<-2',0,'>2'])
cbar.outline.set_visible(False)
axcb.set_ylabel(r'$\Delta$ R / $R_0$', labelpad = 0)

# plot the weights
pcs = transientR['PCA']['neuronWeights']
# normalize by max for each group
rank = np.arange(0, len(pcs))
for i in range(np.min([3,pcs.shape[1]])):
    y= pcs[:,i]
    ax2.fill_betweenx(rank, np.zeros(noNeurons),y[results['neuronOrderPCA']], step='pre',\
    alpha=1.0-i*0.2, color=Ls[i])
    
ax2.set_xlabel('Neuron weights')
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_yticks([])
ax2.set_xticks([])
# plot PCA components
for i in range(np.min([len(results['pcaComponents']), 3])):
    y = results['pcaComponents'][i]
    # normalize
    y =y -np.min(y)
    y =y/np.max(y)
    ax3.text(-150, np.max(y)+i*1.1, 'PC{}'.format(i+1), color = Ls[i])
    ax3.plot(time, i*1.1+y, label='Component {}'.format(i+1), lw=1, color = Ls[i])
#ax3.legend()
#ax4.set_ylabel('PCA components')
ax3.set_xlabel('Time (s)')
ax3.set_xlim([np.min(time), np.max(time)])
ax3.spines['left'].set_visible(False)
ax3.set_yticks([])
# plot dimensionality for inactive and active plus together
nComp = 10#results['nComp']
for y, col in zip([results['expVariance'][:nComp],resultshalf['expVariance'][:nComp], results2half['expVariance'][:nComp]], ['k', R1, B1]):
    ax4.fill_between(np.arange(0.5,nComp+0.5),y*100, step='post', color=col, alpha=0.5)
    ax4.plot(np.arange(1,nComp+1),np.cumsum(y)*100, 'o-',color = col, lw=1, markersize =3) 

ax4.set_ylabel('Variance explained (%)')
ax4.set_yticks([0,25,50,75,100])
ax4.set_xlabel('Number of \n components')
ax4.set_xticks([0,5, 10])

################################################
#
# second row
#
################################################

ax5 = plt.subplot(gs1[1,0], projection = '3d')

# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['E', 'F', 'G']
y0 = 0.68
locations = [(0,y0), (0.35,y0), (0.76,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
            horizontalalignment='left',verticalalignment='top',)

# plot manifold for split dataset
x,y,z = results['pcaComponents'][:3]
#x = x/np.max(x)
#y = y/np.max(x)
#z = z/np.max(x)

# make smoooth
smooth = 12
x = gaussian_filter1d(x, smooth)
y = gaussian_filter1d(y, smooth)
z = gaussian_filter1d(z, smooth)
# color by before and after
colorBy = np.zeros(len(time))
colorBy[:6*60*4] = 1 # first four minutes is m9
multicolor(ax5,x,y,z,colorBy,c= transientcmap, threedim = True, etho = False, cg = 1)
ax5.scatter3D(x[::12], y[::12], z[::12], c=colorBy[::12], cmap=transientcmap, s=10)
ax5.view_init(elev=40, azim=55)
ax5.dist = 7.5
axmin, axmax = -0.04, 0.04
ticks = [axmin,0, axmax]
#plt.setp(ax5.get_xticklabels(), fontsize=10)
#plt.setp(ax5.get_yticklabels(), fontsize=10)
#plt.setp(ax5.get_zticklabels(), fontsize=10)
ax5.set_xlim([axmin, axmax])
ax5.set_ylim([axmin, axmax])
ax5.set_zlim([axmin, axmax])
#ax5.set_xticks(ticks)
#ax5.set_yticks(ticks)
#ax5.set_zticks(ticks)
ax5.tick_params(axis='both', which='major', pad=0)
ax5.axes.xaxis.set_ticklabels([])
ax5.axes.yaxis.set_ticklabels([])
ax5.axes.zaxis.set_ticklabels([])
#ax5.set_xlabel('\nPC1', fontsize=10)
#ax5.set_ylabel('\nPC2', fontsize=10)
#ax5.set_zlabel('\nPC3', labelpad =3, fontsize=10)
# modify where subplot is
points = ax5.get_position().get_points()
ax5.set_position(mpl.transforms.Bbox(ax5.get_position()))
# make scalebar
axesNames = [ax5.xaxis, ax5.yaxis, ax5.zaxis]
for tmp, loc in zip(axesNames, [(0,0,0),(1,1,1),(2,2,2)]):
    tmp._axinfo['juggled']=loc

# make a scale bar in 3d
scX, scY, scZ = 0.02,0.005,-0.04
names = ['PC1', 'PC2', 'PC3']
align = ['right', 'left','center']
for i in range(3):
    l = np.zeros(3)
    l[i] = 0.02
    ax5.plot([scX, scX +l[0]], [scY, scY+l[1]], [scZ, scZ+l[2]], color='k')
    l = np.zeros(3)+axmin
    l[i] = axmax+0.0075
    ax5.text(l[0], l[1], l[2], names[i], horizontalalignment=align[i],\
        verticalalignment='center')
pos = ax5.get_position().get_points()
pos[:,0] -=0.035
pos[:,1] -=0.035
posNew = mpl.transforms.Bbox(pos)
ax5.set_position(posNew)
# show projections
gsProjections = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs1[1,1:3], wspace=1)

axXY = plt.subplot(gsProjections[0,0])
axXZ = plt.subplot(gsProjections[0,1])
axZY = plt.subplot(gsProjections[0,2])

multicolor(axXY,x,y,None,colorBy,c=transientcmap, threedim = False, etho = False, cg = 1)
axXY.set_xlabel('PC1', labelpad=0)
axXY.set_ylabel('PC2', labelpad=0)

multicolor(axXZ,x,z,None,colorBy,c=transientcmap, threedim = False, etho = False, cg = 1)
axXZ.set_xlabel('PC1')
axXZ.set_ylabel('PC3')
#
multicolor(axZY,z,y,None,colorBy,c=transientcmap, threedim = False, etho = False, cg = 1)
axZY.set_xlabel('PC3')
axZY.set_ylabel('PC2')
axZY.set_xticks([-0.05,0, 0.05])
axZY.set_yticks([-0.05, 0, 0.05])
for ax in [axXY, axXZ, axZY]:
    ax.set_xticks([-0.05,0, 0.05])
    ax.set_yticks([-0.05,0, 0.05])
    plt.setp(ax.get_xticklabels(), rotation=-25)


# plot rank of neuron in first vs second half
ax6 = plt.subplot(gs1[1,3])
for pc in range(3):
    rankHalf1, rankHalf2 = np.argsort(resultshalf['neuronWeights'][:,pc]),  np.argsort(results2half['neuronWeights'][:,pc])
    ax6.scatter(rankHalf1, rankHalf2, alpha=0.75, s = 5,color = Vs[pc+1] )
    print 'R2', np.corrcoef(rankHalf1, rankHalf2)
ax6.set_xlabel('Weights moving')
ax6.set_ylabel('Weights immobilized')

################################################
#
#third row
#
################################################    
# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['H', 'I', 'J', 'K']
y0 = 0.3
locations = [(0,y0), (0.35,y0), (0.76,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
            horizontalalignment='left',verticalalignment='top',)

# Todo variance explained for moving and immobile
colorsExp = {'moving': R1, 'immobilized': B1}
colorCtrl = {'moving': N0,'immobilized': N1}
gsExpVar = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs1[2,0], wspace=0.5)
ax11 = plt.subplot(gsExpVar[0,0])
ax12 = plt.subplot(gsExpVar[0,1])
for typ, colors, ax in zip(['AML32', 'AML18'], [colorsExp, colorCtrl], [ax11, ax12]):
    for condition in ['moving', 'immobilized']:
        key = '{}_{}'.format(typ, condition)
        dset = data[key]['analysis']
        tmpdata = []
        for idn in dset.keys():
            results=  dset[idn]['PCA']
            rescale=  data[key]['input'][idn]['Neurons']['Activity'].shape[0]
            tmpdata.append(np.cumsum(results['expVariance'][:nComp])*100)       
        ax.plot(np.arange(1,nComp+1),np.mean(tmpdata, axis=0) ,'o-',color = colors[condition], lw=1, label = '{} {}'.format(typ, condition))
        ax.errorbar(np.arange(1,nComp+1), np.mean(tmpdata, axis=0), np.std(tmpdata, axis=0), color = colors[condition])

ax11.set_ylabel('Explained variance (%)')
ax11.set_yticks([0,25,50,75,100])
ax12.set_yticks([0,25,50,75,100])
ax11.set_xlabel('Number of components')
plt.legend()



plt.show()


#################################
#
# Activity analysis like PNAS paper
#
#################################
gsAct = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs1[2,1:], wspace=0.5)
ax13 = plt.subplot(gsAct[0,0])
ax14 = plt.subplot(gsAct[0,1])
ax15 = plt.subplot(gsAct[0,2])
# extract neural activity histogram for all datasets
bins = np.linspace(-1,2,50)
x = bins[:-1] + np.diff(bins)*0.5
activities = []
for typ in ['AML32', 'AML18']:
    for condition in ['moving', 'immobilized']:
        key = '{}_{}'.format(typ, condition)
        dset = data[key]['input']
        tmpdata = []
        for idn in dset.keys():
            tmpdata.append(np.mean([np.histogram(n[np.isfinite(n)], bins, density=True)[0] for n in dset[idn]['Neurons']['RawActivity']], axis=0))
        activities.append(tmpdata)
# plot gfp and gcamp moving in a panel
histograms = []
for hindex, (hist, c) in enumerate(zip([activities[0],  activities[2]], [colorsExp['moving'], colorCtrl['moving']])):
    m, s = np.nanmean(hist, axis=0), np.nanstd(hist, axis=0)/np.sqrt(len(hist))
    ax13.plot(x, m, color = c, zorder=2)
    ax13.fill_between(x, m-s,m+s, color = c, alpha=0.5)
    histograms.append(m)
# plot gfp and gcamp immobiulized in a panel
for hindex, (hist, c) in enumerate(zip([activities[1],  activities[3]], [colorsExp['immobilized'], colorCtrl['immobilized']])):
    m, s = np.nanmean(hist, axis=0), np.nanstd(hist, axis=0)/np.sqrt(len(hist))
    ax14.plot(x, m, color = c, zorder=2)
    ax14.fill_between(x, m-s,m+s, color = c, alpha=0.5)
    histograms.append(m)

ax13.set_xlabel(r'$\Delta R/R_0$')
ax13.set_xticks([-1,0,1,2])
ax14.set_xticks([-1,0,1,2])
ax14.set_xlabel(r'$\Delta R/R_0$')
ax13.set_ylabel('Normalized Counts')

# plot probability of signal
ax15.plot(x, histograms[0]/(histograms[0]+histograms[1]), color=colorsExp['moving'])
ax15.plot(x, histograms[2]/(histograms[2]+histograms[3]), color=colorsExp['immobilized'])
# draw a line where 95 percent level
ax15.axhline(y = 0.95,color='k', linestyle = '--')
ax15.set_xlabel(r'$\Delta R/R_0$')
ax15.set_ylabel('P(Signal)')
ax15.set_xticks([-1,0,1, 2])
ax15.set_yticks([0.5,1])


# boxplot of signal percentage in each recording
ax12 = plt.subplot(gs1[2,3])
color, labels, ydata = [],[],[]
for typ, colors in zip(['AML32', 'AML18'], [colorsExp, colorCtrl]):
    for condition in ['moving', 'immobilized']:
        color.append(colors[condition])
        labels.append('{} {}'.format(typ, condition))
        tmpdata = []
        key = '{}_{}'.format(typ, condition)
        dset = data[key]['input']
        for idn in dset.keys():
            tmpdata.append(np.mean(dset[idn]['Neurons']['Activity']))
        ydata.append(tmpdata)
x_data = np.arange(len(ydata))
mkStyledBoxplot(ax12, x_data, ydata, color, labels)



# dynamics are slower for immobilized - show autocorrelations 
#ax17 = plt.subplot(gs1[3,1:2])
#for typ, colors, ax in zip(['AML70', 'AML18'], [colorsExp, colorCtrl], [ax11, ax12]):
#    for condition in ['moving', 'immobilized']:
#        key = '{}_{}'.format(typ, condition)
#        dset = data[key]['analysis']
#        tmpdata = []
#        for idn in dset.keys():
#            results=  dset[idn]['Period']
#           
#            tmpdata.append(np.cumsum(results['expVariance'][:nComp])*100*rescale)       
#        ax.plot(np.arange(1,nComp+1),np.mean(tmpdata, axis=0) ,'o-',color = colors[condition], lw=1, label = '{} {}'.format(typ, condition))
#        ax.errorbar(np.arange(1,nComp+1), np.mean(tmpdata, axis=0), np.std(tmpdata, axis=0), color = colors[condition])
#
#ax11.set_ylabel('Explained variance (%)')
#ax11.set_yticks([0,25,50,75,100])
#ax12.set_yticks([0,25,50,75,100])
#ax11.set_xlabel('Number of components')
#plt.legend()

## find highest correlated neuron with velocity
#r2v, r2t = np.max(transientR['CorrelationHalf']['AngleVelocity']), np.max(transientR['CorrelationHalf']['Eigenworm3'])
#veloNeuron = np.argmax(np.abs(transientR['CorrelationHalf']['AngleVelocity']))
#turnNeuron = np.argmax(np.abs(transientR['CorrelationHalf']['Eigenworm3']))
## create an ethogram based on this for the whole recording
#gsN = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs1[2,0], height_ratios=[0.5,1,1])
#
#ax7 = plt.subplot(gsN[0,0])
#ax8 = plt.subplot(gsN[1,0])
#ax9 = plt.subplot(gsN[2,0])
#
#mp.plotEthogram(ax7, time, transient['Behavior']['Ethogram'], alpha = 1, yValMax=1, yValMin=0, legend=0)
#ax7.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
#ax7.set_xticks([])
#ax7.xaxis.label.set_visible(False)
#ax8.plot(time, transient['Neurons']['Activity'][veloNeuron], color=R2, label = r'$R^2=${:.2f}'.format(r2v))
#ax8.set_xticks([])
#ax8.legend()
#ax9.plot(time, transient['Neurons']['Activity'][turnNeuron], color=B2, label = r'$R^2=${:.2f}'.format(r2t))
#ax9.set_xlabel('Time (s)')
#ax9.legend()