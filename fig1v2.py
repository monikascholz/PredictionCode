
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

################################################
#
# create figure 1: This is twice the normal size
#
################################################

fig = plt.figure('Fig - 1 : Neural dynamics in freely moving animals', figsize=(9.5, 9*3/4.))
gsHeatmap = gridspec.GridSpec(5,4,  width_ratios=[1.5, 0.1, 0.5, 0.5], height_ratios = [1,0.1,0.75,0.1, 0.75])
gsHeatmap.update(left=0.075, right=0.98,  bottom = 0.11, top=0.98, hspace=0.3, wspace=0.45)
#fig.patch.set_alpha(0.0)
#heatmap axes
axhm = plt.subplot(gsHeatmap[0,0])
axcb = plt.subplot(gsHeatmap[0,1])
# ethogram
axetho = plt.subplot(gsHeatmap[1,0], clip_on=False)
# legend for ethogram
axEthoLeg = plt.subplot(gsHeatmap[1:2,1])#,clip_on=False)
# principal components
ax4 =plt.subplot(gsHeatmap[2:3,0], clip_on=False)#, sharex=axhm)
# subpanel layout for autocorr
gsPer= gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gsHeatmap[:3,2:], wspace=0.35,hspace=0.2, height_ratios=[1,1.25])
ax13 = plt.subplot(gsPer[0])
ax14 = plt.subplot(gsPer[1])
# manifooolds
ax5 = plt.subplot(gsPer[2], projection='3d', clip_on = False, zorder=-10, aspect='equal')
ax6 = plt.subplot(gsPer[3], projection='3d', clip_on = False, zorder=-10, aspect='equal')
# rank correlations
gsRank = gridspec.GridSpecFromSubplotSpec(1,6,subplot_spec=gsHeatmap[4,:], width_ratios=[1.0,1,1,1,0.1,1.0], wspace=0.2)

ax11 = plt.subplot(gsRank[5])
ax8 = plt.subplot(gsRank[1])
ax9 = plt.subplot(gsRank[2])
ax10 = plt.subplot(gsRank[3])
axexpV = plt.subplot(gsRank[0])
#axcbar2 = plt.subplot(gsHeatmap[4,1])
axcbar2 = plt.subplot(gsRank[4])
# scale last plot so it aligns with autocorrelations
alignAxes(ax14, ax11, where='xspan')

#plt.show()
################################################
#
# letters
#
################################################
# mark locations on the figure to get good guess for a,b,c locs
#for y in np.arange(0,1.1,0.1):
#    plt.figtext(0, y, y)
#for x in np.arange(0,1.1,0.1):
#    plt.figtext(x, 0.95, x)
#
#letters = map(chr, range(65, 91)) 
## add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B', 'C']
x0 = 0
locations = [(x0,0.97),  (x0,0.665), (x0,0.58)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
#
letters = ['D', 'E']
x0 = 0.57
locations = [(x0,0.97),  (x0,0.665)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)

letters = ['F','G', 'H']
y0 = 0.3
locations = [(0,y0),  (0.24,y0), (0.77,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
#
#letters = ['I', 'J']
#y0 = 0.27
#locations = [(0,y0),  (0.22,y0), (0.76,y0)]
#for letter, loc in zip(letters, locations):
#    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
#            horizontalalignment='left',verticalalignment='top',)
################################################
#
#first row
#
################################################
# select a special dataset - transiently immobilized
transientData = 'BrainScanner20180511_134913'
transient = data['Special_transition']['input']['BrainScanner20180511_134913']
transientAnalysis = data['Special_transition']['analysis']['BrainScanner20180511_134913']
# time first half, second half. Indices of times
timeHalf = np.arange(0, 1400)
time2Half = np.arange(1600, transient['Neurons']['Activity'].shape[1])
# pull out repeated stuff
time = transient['Neurons']['TimeFull']
timeActual = transient['Neurons']['Time']
noNeurons = transient['Neurons']['Activity'].shape[0]
results = transientAnalysis['PCA']
resultshalf = transientAnalysis['PCAHalf1']
results2half = transientAnalysis['PCAHalf2']
test, train = transientAnalysis['Training']['Half']['Test'], transientAnalysis['Training']['Half']['Train']
colorsExp = {'moving': R1, 'immobilized': B1}
colorsCtrl = {'moving': N0,'immobilized': N1}
         
#heatmap
cax1 = plotHeatmap(time, transient['Neurons']['ActivityFull'][results2half['neuronOrderPCA']], ax=axhm, vmin=-0.5, vmax=2)
axhm.xaxis.label.set_visible(False)
axhm.set_xticks([])
# colorbar
cbar = fig.colorbar(cax1, cax=axcb, use_gridspec = True)
cbar.set_ticks([-0.5,0,2])
cbar.set_ticklabels(['<-0.5',0,'>2'])
cbar.outline.set_visible(False)
moveAxes(axcb, 'left', 0.06)
moveAxes(axcb, 'scaley', -0.08)
axcb.set_ylabel(r'$\Delta R/R_0$', labelpad = -25)
#ethogram

plotEthogram(axetho, time, transient['Behavior']['EthogramFull'], alpha = 1, yValMax=1, yValMin=0, legend=0)
cleanAxes(axetho, 'all')
moveAxes(axetho, 'scaley', 0.02)
axetho.xaxis.label.set_visible(False)
# legend for ethogram
#moveAxes(axEthoLeg, 'right', 0.025)
moveAxes(axEthoLeg, 'up', 0.02)
cleanAxes(axEthoLeg, where='all')

handles, labels = axetho.get_legend_handles_labels()
leg = mpl.legend.Legend(axEthoLeg, handles[::-1], labels[::-1],frameon=1, loc=1,prop={'size':12},handlelength=0.5, labelspacing=0,handletextpad=0.5)#,bbox_to_anchor=(-1, 0.9), loc=9)
for hndl in leg.legendHandles:
    hndl._sizes = [0]
axEthoLeg.add_artist(leg);

# plot mean autocorrelation moving versus immobile
movExp = ['AML32_moving', 'AML70_chip']
imExp = ['AML32_immobilized', 'AML70_immobilized']

# all moving gfp
movCtrl = ['AML18_moving', 'AML175_moving']
imCtrl = ['AML18_immobilized']
# color coding
colorsExp = {'moving': R1, 'immobilized': B1}
colorCtrl = {'moving': N0,'immobilized': N1}



typ, colors, axes = [movExp, imExp], colorsExp, [ax13, ax14]
for  condition, ax, keys in zip(['moving', 'immobilized'], axes, typ):
    tmpdata = []
    for key in keys:
        #key = '{}_{}'.format(typ, condition)
        dset = data[key]['analysis']
        for idn in dset.keys():
            tmpdata.append(np.mean(dset[idn]['Period']['NeuronACorr'],axis=0)) 
            T = dset[idn]['Period']['Periods']
    m, s = np.nanmean(tmpdata, axis=0), np.nanstd(tmpdata, axis=0)
    tmpdata= np.array(tmpdata)
    ax.plot(T,tmpdata.T ,'-',color = colors[condition], lw=1.5, alpha=0.35,label = '{} {}'.format(typ, condition))
    if typ==imExp:
        ax.plot(T,tmpdata[0] ,'-',color = colors[condition], lw=2, alpha=1,label = '{} {}'.format(typ, condition))
    #ax.plot(T,np.mean(tmpdata,axis=0) ,'-',color = colors[condition], lw=5, alpha=0.5,label = '{} {}'.format(typ, condition))
    #ax.fill_between(dset[idn]['Period']['Periods'], m-s, m+s, alpha=0.5, zorder=-1,color = colors[condition])
    ax.axhline(color='k', linestyle = '--', zorder=-1)
    ax.set_ylim([-0.2,1])
    ax.text(0.5, 0.9,condition, transform=ax.transAxes, horizontalalignment='center')
    ax.set_xticks([0,150,300])
ax13.set_ylabel('Autocorrelation')
#ax13.text(-0.5,0,'Autocorrelation', fontsize=14,transform = ax13.transAxes, rotation=90, verticalalignment ='center')
ax13.set_xlabel('Lag (s)')
ax14.set_xlabel('Lag (s)')
ax14.set_yticks([])
#ax16.set_yticks([])

# plot autocorr of our favorite dataset
ax13.plot(transientAnalysis['Period1Half']['Periods'],np.mean(transientAnalysis['Period1Half']['NeuronACorr'],axis=0), colorsExp['moving'], lw=1.5)
ax14.plot(transientAnalysis['Period2Half']['Periods'],np.mean(transientAnalysis['Period2Half']['NeuronACorr'],axis=0), colorsExp['immobilized'], lw=1.5)

# plot dimensionality for inactive and active plus together
nComp = 10#results['nComp']
for y, col, lab, mark in zip([resultshalf['expVariance'][:nComp], results2half['expVariance'][:nComp]]\
        , [ R1, B1], [ 'Moving', 'Paralyzed'], [ '^', 's']):
    #ax3.fill_between(np.arange(0.5,nComp+0.5),y*100, step='post', color=col, alpha=0.5)
   axexpV.plot(np.arange(1,nComp+1),np.cumsum(y)*100, 'o-',color = col, label = lab, marker=mark) 

axexpV.set_ylabel('Variance exp. (%)', labelpad=-5)
axexpV.set_yticks([0,25,50,75,100])
axexpV.set_xlabel('# of components')
axexpV.set_xticks([0,5, 10])

# # Todo variance explained for moving and immobile -- should be in supplementary

#ax11 = plt.subplot(gsHeatmap[2,-1])
#for condition, keys, mark in zip([ 'immobilized','moving'], [ imExp,movExp], ['s', '^']):
#    for key in keys:
#        dset = data[key]['analysis']
#        tmpdata = []
#        for idn in dset.keys():
#            results=  dset[idn]['PCA']
#            rescale=  data[key]['input'][idn]['Neurons']['Activity'].shape[0]
#            tmpdata.append(np.cumsum(results['expVariance'][:nComp]*100))       
#    ax11.plot(np.arange(1,nComp+1),np.mean(tmpdata, axis=0) ,'-',color =colorsExp[condition], lw=1, label = '{} {}'.format(typ, condition))
#    ax11.errorbar(np.arange(1,nComp+1), np.mean(tmpdata, axis=0), np.std(tmpdata, axis=0), color = colorsExp[condition], marker=mark)
#
##ax11.set_ylabel('Explained variance (%)')
#ax11.set_yticks([0,25,50,75,100])
##ax12.set_yticks([0,25,50,75,100])
#ax11.set_xlabel('# of components')
#plt.legend()

#################################################
##
## second row
##
#################################################

# plot PCA components
for i in range(np.min([len(results2half['pcaComponents']), 3])):
    #y = results['pcaComponents'][i]
    y = results2half['fullData'][i]
    # normalize
    y =y -np.min(y)
    y =y/np.max(y)
    ax4.text(-100, np.mean(y)+i*1.05, 'PC{}'.format(i+1), color = 'k')
    ax4.plot(time[transient['Neurons']['valid']], i*1.1+y, label='Component {}'.format(i+1), lw=1, color = 'k')

yloc = ax4.get_ylim()[-1]*1.1
## indicate immobilization etc
for label, segment in zip(['moving', 'immobilized'], [train, test]):

    ax4.text(np.mean(timeActual[segment]), 1.02*yloc, label,horizontalalignment='center', color=colorsExp[label])
    ax4.plot([timeActual[segment[0]],timeActual[segment[-1]]], [yloc, yloc], color=colorsExp[label])
# add tetramisole
ax4.text(np.mean(timeActual[train[-1]]), 0.98*yloc, "+ tet",horizontalalignment='left', color='k')
ax4.plot([timeActual[train[-1]],timeActual[test[-1]]], [0.96*yloc, 0.96*yloc], color='k', linestyle='--')
# labels and such
ax4.set_xlabel('Time (s)')
ax4.set_xlim([np.min(timeActual), np.max(timeActual)])
ax4.set_ylim([ax4.get_ylim()[0], yloc*1.01])
cleanAxes(ax4, where='y')
moveAxes(ax4, 'down', 0.02)

# plot manifold! MANIFOOOOOLD!

# plot manifold for split dataset
x,y,z = results2half['fullData'][:3]
x/=np.max(x)
y/=np.max(y)
z/=np.max(z)

# make smoooth
smooth = 12
x = gaussian_filter1d(x, smooth)
y = gaussian_filter1d(y, smooth)
z = gaussian_filter1d(z, smooth)
# color by before and after
colorBy = np.zeros(len(timeActual))
colorBy[train] = 1 # first four minutes is m9
ax5.plot(x[train],y[train],z[train], color=R1)
ax6.plot(x[test],y[test],z[test], color=B1)
ax5.scatter(x[train[::12]],y[train[::12]],z[train[::12]], color=R1, s=5)
ax6.scatter(x[test[::12]],y[test[::12]],z[test[::12]], color=B1, s=5)

for ax in [ax5, ax6]:    
    ax.view_init(elev=40, azim=150)
#        ax.dist = 7
    axmin, axmax = -1,0.9
#        ticks = [axmin,0, axmax]
    
#        ax.set_xlim([axmin, axmax])
#        ax.set_ylim([axmin/2., axmax])
#        ax.set_zlim([axmin, axmax])
#        #
#        ax.tick_params(axis='both', which='major', pad=-10)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
#        #make scalebar
    axesNames = [ax.xaxis, ax.yaxis, ax.zaxis]
    for tmp, loc in zip(axesNames, [(0,0,0),(1,1,1),(2,2,2)]):
        tmp._axinfo['juggled']=loc

    ax.set_xlim([axmin, axmax])
    ax.set_ylim([axmin/2., axmax])
    ax.set_zlim([axmin, axmax])


    moveAxes(ax, action='left', step=0.025 )
    moveAxes(ax5, action='left', step=0.01 )
    moveAxes(ax, action='down', step=0.03 )
    moveAxes(ax, action='scale', step=0.10 )
#        # make a scale bar in 3d
scX, scY, scZ = -1,2,0
names = [r'PC$_1$', 'PC$_2$', 'PC$_3$']
align = ['center', 'right','center']
aligny = [ 'top','center','baseline','center']
for i in range(3):
    l = np.zeros(3)
    l[i] = 0.25
    ax.plot([scX, scX -l[0]], [scY, scY+l[1]], [scZ, scZ+l[2]], color='k', clip_on=False)
    ax.text(scX -l[0], scY+l[1], scZ+l[2], names[i],color='k',fontsize=12, \
    horizontalalignment=align[i], verticalalignment=aligny[i])
    #l = np.zeros(3)#+axmin


# pc axes projection
#sciformat = 1.
#multicolor(axproj,x*sciformat,y*sciformat,None,colorBy,c=transientcmap, threedim = False, etho = False, cg = 1)
##axproj.set_xlabel(r'PC1 ($\times \, 10^{-2}$)', labelpad=0, color=Ls[0])
##axproj.set_ylabel(r'PC2 ($\times \, 10^{-2}$)', labelpad=0, color=Ls[1])
#axproj.set_xlabel(r'PC1', labelpad=0, color=Ls[0])
#axproj.set_ylabel(r'PC2', labelpad=0, color=Ls[1])
##moveAxes(axproj, action='up', step=0.02 )
#

#################################################
##
## third row - autocorrelations
##
#################################################
# individual rank order correlations for three examples


movData = 'BrainScanner20170613_134800'
immData = 'BrainScanner20180510_092218'
transientData = 'BrainScanner20180511_134913'
for key, dset, label, ax in zip(['AML32_moving', 'AML32_immobilized', 'AML32_chip'],[movData, immData, transientData], ['moving', 'immobilized', 'transient'], [ax8, ax9, ax10]):
    rankC = np.abs(np.copy(data[key]['analysis'][dset]['PCArankCorr']))
    cax1 = ax.imshow(rankC, vmin=0, vmax=0.5,origin='lower')
    ax.set_xticks(np.arange(0,3,1))
    ax.set_yticks(np.arange(0,3,1))
    ax.set_xticklabels([r'$PC_{11}$',r'$PC_{21}$',r'$PC_{31}$'])
    ax.set_yticklabels([])
    for i in range(3):
        # find best match, delete that option
        ym, xm = np.unravel_index(rankC.argmax(), rankC.shape)
        if np.max(rankC)>0.25:
            c = 'k'
        else:
            c='w'
        ax.text(xm,ym, np.round(np.max(rankC), decimals=2), color=c, horizontalalignment ='center', verticalalignment ='center')
        print np.max(rankC)        
        rankC[ym,:] =0
        rankC[:,xm] =0
#        plt.imshow(rankC)
#        plt.show()
#        print np.max(rankC)
#        print xm, ym
#    for index, entry in enumerate(np.ravel(rankC)):
#        loc = np.unravel_index(index, rankC.shape)
#        ax.text(loc[0],loc[1], np.round(entry, decimals=2), horizontalalignment ='center', verticalalignment ='center')
    ax.set_title(label)
ax8.set_yticklabels([r'$PC_{12}$',r'$PC_{22}$',r'$PC_{32}$'])
moveAxes(ax8, 'right', 0.04)
moveAxes(ax10, 'left', 0.04)
for ax in [ax8, ax9, ax10]:
    moveAxes(ax, 'scale', -0.04)
    moveAxes(ax, 'left', 0.025)

    alignAxes(ax11, ax, 'y')

alignAxes(ax8, axcbar2, 'yspan')
# colorbar for rank correlations
cbar = fig.colorbar(cax1, cax = axcbar2, ax = [ax8, ax9, ax10], use_gridspec = True)

cbar.set_ticks([0,0.5])
cbar.set_ticklabels(['0', '>0.5'])
cbar.outline.set_visible(False)
moveAxes(axcbar2, 'scaley', -0.05)
moveAxes(axcbar2, 'left', 0.08)
axcbar2.set_ylabel(r'Correlation', labelpad = 0)
axcbar2.yaxis.set_label_position('left')
#for ax in [ax8, ax9, ax10, axcbar2]:
#    moveAxes(ax, 'down', 0.05)
#

    
markers = {'AML32_moving': 'o','AML70_chip':'^' ,'AML32_immobilized':'o', 'AML70_immobilized':'^'}
x0=0.5
locs = {'AML32_moving': x0,'AML70_chip':x0 ,'AML32_immobilized':x0+1, 'AML70_immobilized':x0+1}
colors = {'AML32_moving': R1,'AML70_chip':R1 ,'AML32_immobilized':B1, 'AML70_immobilized':B1}
# rank order of PCA weights

boxplot = []
for keys in [movExp, imExp]:
    r2 = []
    for key in keys:
        keep = []
        dset = data[key]['analysis']
        for idn in dset.keys():
            
            rankC = np.abs(dset[idn]['PCArankCorr'])
            
            tmp = []
            for i in range(3):
                # find best match, delete that option
                tmp.append(np.max(rankC))
                xm, ym = np.unravel_index(rankC.argmax(), rankC.shape)
                rankC[ym,:] =0
                rankC[:,xm] =0
            r2.append( np.mean(tmp))
            keep.append( np.mean(tmp))
        rnd1 = np.random.rand(len(keep))*0.2
        ax11.scatter(np.zeros(len(keep))+rnd1+locs[key]+0.15, keep, marker = markers[key],s=15, c = colors[key], alpha=0.5)
        
    boxplot.append(r2)

# add two transition datasets
dset = data['Special_transition']['analysis']
r2 = []
for idn in dset.keys(): 
    rankC = np.abs(dset[idn]['PCArankCorr'])
    
    tmp = []
    for i in range(3):
        # find best match, delete that option
        tmp.append(np.max(rankC))
        xm, ym = np.unravel_index(rankC.argmax(), rankC.shape)
        rankC[ym,:] =0
        rankC[:,xm] =0
    
    r2.append( np.mean(tmp))
rnd1 = np.random.rand(len(r2))*0.2
ax11.scatter(np.zeros(len(r2))+rnd1+3, r2, marker = 'o',s=15, c = N0, alpha=0.5)
    
boxplot.append(r2)

mkStyledBoxplot(ax11,[1,2, 3], np.array(boxplot), [R1, B1, N0], ['moving', 'immobilized', 'transient'], scatter=False)
ax11.set_ylim([0,1])
ax11.set_yticks([0,0.5, 1])
ax11.set_xlim([0.5,3.5])
ax11.set_ylabel('Correlation')
#### move everything left
#for axi in [ax5, ax13, ax11]:
#    moveAxes(axi, 'left', 0.03)
#
#for axind, ax in enumerate(fig.get_axes()):

plt.show()

############GFP autocorr
#ax15 = plt.subplot(gsPer[2])
#ax16 = plt.subplot(gsPer[3])
#for typ, colors, axes in zip([[movExp, imExp], [movCtrl, imCtrl]], [colorsExp, colorCtrl], [[ax13, ax14], [ax13,ax14]]):
#    for  condition, ax, keys in zip(['moving', 'immobilized'], axes, typ):
#        tmpdata = []
#        for key in keys:
#            #key = '{}_{}'.format(typ, condition)
#            dset = data[key]['analysis']
#            
#            for idn in dset.keys():
#                
#                tmpdata.append(np.mean(dset[idn]['Period']['NeuronACorr'],axis=0)) 
#                T = dset[idn]['Period']['Periods']
#        m, s = np.nanmean(tmpdata, axis=0), np.nanstd(tmpdata, axis=0)
#        tmpdata= np.array(tmpdata)
#        ax.plot(T,tmpdata.T ,'-',color = colors[condition], lw=1.5, alpha=0.35,label = '{} {}'.format(typ, condition))
#        if typ==imExp:
#            ax.plot(T,tmpdata[0] ,'-',color = colors[condition], lw=2, alpha=1,label = '{} {}'.format(typ, condition))
#        #ax.plot(T,np.mean(tmpdata,axis=0) ,'-',color = colors[condition], lw=5, alpha=0.5,label = '{} {}'.format(typ, condition))
#        #ax.fill_between(dset[idn]['Period']['Periods'], m-s, m+s, alpha=0.5, zorder=-1,color = colors[condition])
#        ax.axhline(color='k', linestyle = '--', zorder=-1)
#        ax.set_ylim([-0.2,1])
#        ax.text(0.5, 0.9,condition, transform=ax.transAxes, horizontalalignment='center')
#ax13.set_ylabel('Autocorrelation')


        #ax.imshow( tmpdata, aspect='auto', interpolation='none', origin='lower',extent=[T[0],T[-1],len(tmpdata),0],vmax=1)
#ax13.text(-0.25,0,'Autocorrelation', fontsize=14,transform = ax13.transAxes, rotation=90, verticalalignment ='center')

        
#ax11.set_ylabel('Explained variance (%)')
#ax11.set_yticks([0,25,50,75,100])
#ax12.set_yticks([0,25,50,75,100])
#ax11.text(1.2,-0.3,'Number of components', transform = ax11.transAxes, horizontalalignment ='center')



#gsEWs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gsHeatmap[3,0], wspace=0.1, hspace=0.2, height_ratios=[1,1,1])
## plot eigenworm schematic
#ax6 = plt.subplot(gsEWs[2,0],  adjustable='box')
#ax6.axis('equal')
## plot eigenworms
#eigenworms = dh.loadEigenBasis(filename = 'utility/Eigenworms.dat', nComp=3, new=True)
#lengths = np.array([4])
#meanAngle = np.array([-0.07945])
#refPoint = np.array([[0,0]])
#prefactor = [r'',r'$=a_1$ *',r'+ $a_2$ *',r'+ $a_3$ *' ][::-1]
#descr = ['posture', 'undulation', 'undulation', 'turn'][::-1]
#colors = [B1, B1, R1]
## from which volume to show the posture
#tindex = 500
#for i in range(4):
#    pcs = np.zeros(3)
#    if i ==3:
#        pcsNew, meanAngle, _, refPoint = dh.calculateEigenwormsFromCL(moving['CL'], eigenworms)
#        pc3New, pc2New, pc1New = pcsNew
#        cl = dh.calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint)
#        
#        new = createWorm(cl[tindex,:,0], cl[tindex,:,1])
#    else:
#        pcs[i] = 10
#        clNew = dh.calculateCLfromEW(pcs, eigenworms[:3], meanAngle, lengths, refPoint)[0]
#        new = createWorm(clNew[:,0], clNew[:,1])
#    
#    new -=np.mean(new, axis=0)
#    new[:,0] -= i*600 - 800
#    x, y = np.mean(new, axis =0)
#    p2 = mpl.patches.Polygon(new, closed=True, fc=N0, ec='none')
#    plt.text(x,y-200, descr[i], fontsize=12, horizontalalignment = 'center')
#    plt.text(x-250,y, prefactor[i], fontsize=12, horizontalalignment = 'center')
#    ax6.add_patch(p2)
#    ax6.axes.get_xaxis().set_visible(False)
#    ax6.axes.get_yaxis().set_visible(False)
#    ax6.spines['left'].set_visible(False)
#    ax6.spines['bottom'].set_visible(False)
#    ax6.set_xlim(-1200, 1000)
#    ax6.set_ylim(-50, 10)
#
#
## plot angle velocity and turns
#ax7 = plt.subplot(gsEWs[0,0],sharex=axhm )
#ax7.plot(timeActual, moving['Behavior']['AngleVelocity'], color = R1)
## draw a box for the testset
#ax7.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N2, zorder=-10, alpha=0.75)
#ax7.axhline(color='k', linestyle = '--', zorder=-1)
#ax7.text(np.mean(timeActual[test]), ax4.get_ylim()[-1], 'Testset',horizontalalignment='center')
#ax7.set_ylabel('Wave speed')
## make scalebar
#xscale = timeActual[0]-20
#yscale =  [-0.025, 0.025]
#ax7.plot([xscale, xscale], yscale, color=R1, clip_on=False)
#ax7.text(xscale, np.max(ax7.get_ylim())*1.1, 'Wave speed', color=R1,horizontalalignment='center',verticalalignment='center')
#ax7.text(xscale, 0, np.ptp(yscale), color=R1, rotation = 90,horizontalalignment='right',verticalalignment='center')
#ax7.axes.get_yaxis().set_visible(False)
#ax7.spines['left'].set_visible(False)
## remove xlabels
#plt.setp(ax7.get_xticklabels(), visible=False)
#ax8 = plt.subplot(gsEWs[1,0], sharex=axhm)
#ax8.plot(timeActual,moving['Behavior']['Eigenworm3'], color = B1)
## draw a box for the testset
#ax8.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N2, zorder=-10, alpha=0.75)
##ax8.text(np.mean(timeActual[test]), ax4.get_ylim()[-1], 'Testset',horizontalalignment='center')
#ax8.axhline(color='k', linestyle ='--', zorder=-1)
#ax8.set_ylabel('Turn')
#ax8.set_xlabel('Time (s)')
#ax8.set_xlim([timeActual[0], timeActual[-1]])
## make scalebar
#xscale = timeActual[0]-20
#yscale =  [-7.5, 7.5]
#ax8.plot([xscale, xscale], yscale, color=B1, clip_on=False)
#ax8.text(xscale, np.max(ax8.get_ylim()), 'Turns', color=B1,horizontalalignment='center',verticalalignment='center')
#ax8.text(xscale, 0, int(np.round(np.ptp(yscale))), color=B1, rotation = 90,horizontalalignment='right',verticalalignment='center')
#ax8.axes.get_yaxis().set_visible(False)
#ax8.spines['left'].set_visible(False)
## move up all of them
#ax6.set_zorder(-10)
#for axtmp in [ax7, ax8]:
#    pos = axtmp.get_position().get_points()
#    pos[:,1] +=0.01
#    posNew = mpl.transforms.Bbox(pos)
#    axtmp.set_position(posNew)
#     
#pos=ax6.get_position().get_points()
#posNew = mpl.transforms.Bbox(pos)
#ax6.set_position(posNew)
#moveAxes(ax6, action='scale', step=0.05 )
#moveAxes(ax6, action='down', step=0.02 )
## schematic of behavior prediction from PCA
#gsScheme = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gsHeatmap[3,1:],width_ratios=[1.2,1], wspace=0.1, hspace=0.4)
#axscheme1 = plt.subplot(gsScheme[0,0])
#axscheme1.axis('equal')
#axscheme2 = plt.subplot(gsScheme[0,1], sharey=axscheme1)
## input are PCs, let's only show the testset
##
#x,y,z = movingAnalysis['PCA']['pcaComponents'][:3, test]
#t = moving['Neurons']['Time'][test]
#scale = np.ptp(t)*0.8
#ylocs = np.linspace(0,scale,3)
## for weight circles
#ycircle1 = ylocs +scale/12.
#ycircle2 = ylocs -scale/12.
#xcirc = t[-1]+scale/7
#for lindex, line in enumerate([x,y,z]):
#    line -=np.mean(line)
#    line /=np.max(line)
#    line*=scale/7.
#    axscheme1.plot(t,line+ylocs[lindex], color='k')
#    axscheme1.text(t[0], ylocs[lindex]+scale/7, 'PC{}'.format(lindex+1), horizontalalignment='center')
#    # circles for weights - red
#    circ = mpl.patches.Circle((xcirc, ycircle1[lindex]), scale/10.,fill=True,color='w',lw=2, ec=R1, clip_on=False)
#    axscheme1.text(xcirc, ycircle1[lindex], r'$w_{}$'.format(lindex+1),color=R1, verticalalignment='center', horizontalalignment='center')
#    axscheme1.add_patch(circ)
#    #blue circles
#    circ = mpl.patches.Circle((xcirc, ycircle2[lindex]), scale/10.,fill=True,linestyle=(0, (1, 1)),color='w',lw=2, ec=B1, clip_on=False)
#    axscheme1.text(xcirc, ycircle2[lindex], r'$w_{}$'.format(lindex+1),color=B1, verticalalignment='center', horizontalalignment='center')
#    axscheme1.add_patch(circ)
#    
#ybeh = [ylocs[1]+scale/10., ylocs[0]-scale/10.]+np.diff(ylocs)/2.
#for behavior, color, cpred, yl, label in zip(['AngleVelocity','Eigenworm3' ], \
#            [N1, N1], [R1, B1], ybeh, ['Wave speed', 'Turn']):
#    beh = moving['Behavior'][behavior][test]
#    meanb, maxb = np.mean(beh),np.std(beh)
#    beh = (beh-meanb)/maxb
#    beh*=scale/10
#    behPred = movingAnalysis['PCAPred'][behavior]['output'][test]
##    behPred = (behPred-meanb)/maxb
#    behPred*=scale/10
#    axscheme2.plot(t, beh+yl, color=color)
#    axscheme2.plot(t, behPred+yl, color=cpred)
#    axscheme2.text(t[-1], yl+scale/5, \
#    r'$R^2 = {:.2f}$'.format(np.float(movingAnalysis['PCAPred'][behavior]['scorepredicted'])), horizontalalignment = 'right')
#    axscheme2.text(t[-1]*1.1, yl, label, rotation=90, color=cpred, verticalalignment='center')
##axscheme2.set_zorder(-1)
#
#axscheme2.set_facecolor('none')
#for i in range(3):
#    con = mpl.patches.ConnectionPatch(xyA=(xcirc,ycircle1[i]), xyB=(t[0],ybeh[0]), coordsA="data", coordsB="data",
#                          axesA=axscheme1, axesB=axscheme2, color=R1)
#    axscheme1.add_artist(con)
#    con.set_zorder(-10)    
#    con = mpl.patches.ConnectionPatch(xyA=(xcirc,ycircle2[i]), xyB=(t[0], ybeh[1]), coordsA="data", coordsB="data",
#                          axesA=axscheme1, axesB=axscheme2, color=B1, lw=2, linestyle=':')
#    axscheme1.add_artist(con)
#    con.set_zorder(-10)
## add scalebar
#l =120
#y = ylocs[0] - scale/4.
#axscheme1.plot([t[0], t[0]+l],[y, y], 'k', lw=2)
#axscheme1.text(t[0]+l*0.5,y*0.8, '2 min', horizontalalignment='center')
#axscheme2.plot([t[0], t[0]+l],[y, y], 'k', lw=2)
#axscheme2.text(t[0]+l*0.5,y*0.8, '2 min', horizontalalignment='center')
#
#for axtmp in [axscheme1, axscheme2]:
#    axtmp.spines['left'].set_visible(False)
#    axtmp.spines['bottom'].set_visible(False)
#    axtmp.set_yticks([])
#    axtmp.set_xticks([])
#    #axtmp.set_xlabel('Time (s)')
#
#axscheme1.set_xlim([t[0], 500])
#
#################################################
##
## fourth row
##
#################################################
## predict neural activity from behavior
#gsLasso = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs1[3,0], width_ratios=[1,1], wspace=0.2, hspace=0.1)
#flag = 'PCAPred'
#
#axNav= plt.subplot(gsLasso[0,0])
#axNt = plt.subplot(gsLasso[0,1])
#
#gsBrokenAxis = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsLasso[0,0])
#
#for behavior, colors, axR2 in zip(['AngleVelocity', 'Eigenworm3'], [(R2, N0), (B2, N0)], [axNav, axNt ]):
#
#    alldata = []
#    # experiment
#    c = colors[0]
#    for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
#        dset = data[key]['analysis']
#        keep = []
#        for idn in dset.keys():
#            results=  dset[idn][flag][behavior]
#            keep.append(results['scorepredicted'])
#            
#        keep = np.array(keep)
#        rnd1 = np.random.rand(len(keep))*0.2
#        axR2.scatter(np.zeros(len(keep))+rnd1, keep, marker = marker, c = c, edgecolor=c, alpha=0.5)
#        alldata.append(keep)
#    alldata = np.array(alldata)
#    mkStyledBoxplot(axR2, [-0.5, 1.5], alldata.T, [c], ['GCamp6s'], scatter=False)
#    # controls
#    c = colors[1]
#    ctrldata = []
#    xoff = 1.5
#    for key, marker in zip(['AML18_moving', 'AML175_moving'],['o', "p"]):
#        dset = data[key]['analysis']
#        keep = []
#        for idn in dset.keys():
#            results=  dset[idn][flag][behavior]
#            keep.append(results['scorepredicted'])
#        keep = np.array(keep)
#        rnd1 = np.random.rand(len(keep))*0.2
#        axR2.scatter(xoff+np.zeros(len(keep))+rnd1, keep, marker = marker,c = c, edgecolor=c, alpha=0.5)
#        ctrldata.append(keep)
#    ctrldata = np.array(ctrldata)
#    mkStyledBoxplot(axR2, [-0.5+xoff, 1.5+xoff], ctrldata.T, [c,], ['Control (GFP)'], scatter=False)
#    
#    axR2.set_xlim([-1, 2.5])
#    axR2.set_xticks([-0.5,-0.5+xoff])
#    axR2.set_xticklabels(['GCaMP6s', 'GFP'])
#axNav.set_ylabel(r'$R^2$ (Testset)')

plt.show()
# get all the weights for the different samples

#for typ, colors, ax in zip(['AML32', 'AML18'], [colorsExp, colorCtrl], [ax11, ax12]):
#    for condition in ['moving', 'immobilized']:
#        key = '{}_{}'.format(typ, condition)
#        dset = data[key]['analysis']
#        tmpdata = []
#        for idn in dset.keys():
#            results=  dset[idn]['PCA']
#            rescale=  data[key]['input'][idn]['Neurons']['Activity'].shape[0]
#            tmpdata.append(np.cumsum(results['expVariance'][:nComp])*100)       
#        ax.plot(np.arange(1,nComp+1),np.mean(tmpdata, axis=0) ,'o-',color = colors[condition], lw=1, label = '{} {}'.format(typ, condition))
#        ax.errorbar(np.arange(1,nComp+1), np.mean(tmpdata, axis=0), np.std(tmpdata, axis=0), color = colors[condition])


#ax10 = plt.subplot(gsPred[0,1])
#for li,line in enumerate(orderedWeights):
#    ax10.plot(np.abs(line), label = ('weights for PC{}'.format(li+1)), color='C5', alpha=0.25+0.05*li, lw=1)
#ax10.set_ylabel('Weights')
#
#ax10.plot(np.mean(np.abs(orderedWeights), axis=0), color='C5', alpha=1, lw=2, marker = 'o')
#ax10.set_xticks(np.arange(len(res['behaviorOrder'])))
#ax10.set_xticklabels(res['behaviorLabels'][res['behaviorOrder']])
#
#plt.setp(ax10.get_xticklabels(), rotation=-25)

### plot correlation of PC axes and velocity/turns
#ax9 = plt.subplot(gsHeatmap[3,2])
#color, labels, ydata1, ydata2 = [],[],[], []
#condition = 'moving'
#
#for typ, colors in zip(['AML32', 'AML18'], [colorsExp, colorCtrl]):
#    
#        color.append(colors[condition])
#        labels.append('{} {}'.format(typ, condition))
#        tmpdata1 = []
#        tmpdata2 = []
#        key = '{}_{}'.format(typ, condition)
#        dset = data[key]['analysis']
#        for idn in dset.keys():
#            tmpdata1.append(dset[idn]['PCACorrelation']['AngleVelocity'][:3])
#            tmpdata2.append(dset[idn]['PCACorrelation']['Eigenworm3'][:3])
#        ydata1.append(tmpdata1)
#        ydata2.append(tmpdata2)
#x_data = np.arange(len(ydata1))
#print ydata1
#mp.mkStyledBoxplot(ax9, x_data, ydata1, color, labels)
#mp.mkStyledBoxplot(ax9, x_data+0.5, ydata2, color, labels)
#plt.show()
#reverse prediction


# plot stuff
#plt.figure('PredictedNeuralActivity', figsize=(2.28*4,2.28*6))
#
## show reduced dimensionality heatmap
#mp.plotHeatmap(moving['Neurons']['Time'][test], res['lowDimNeuro'][:,test])
#plt.subplot(322)
#mp.plotHeatmap(moving['Neurons']['Time'][test], newHM[:,test], vmin=np.min(newHM)*1.1, vmax=np.max(newHM)*0.9)
#plt.subplot(324)
#for ind, i in enumerate(res['PCA_indices'][:4]):
#    x = moving['Neurons']['Time'][test]
#    line1, = plt.plot(x, res['NeuralPCS'][test,i]+ind*12, color='C0', label='Neural PCs')
#    line2, = plt.plot(x, res['predictedNeuralPCS'][test,i]+ind*12, color='C3', label= 'Predicted')
#    plt.text(x[-1]*0.9, 1.2*np.max(res['predictedNeuralPCS'][test,i]+ind*10), '$R^2={:.2f}$'.format(res['R2_test'][ind]))
#plt.legend([line1, line2], ['Neural PCs', 'Predicted from Behavior'], loc=2)
#ylabels = ['PC {}'.format(index+1) for index in res['PCA_indices'][:4]]
#plt.yticks(np.arange(0,4*12, 12), ylabels)
#plt.xlabel('Time(s)')
#plt.subplot(323)
#for ind, i in enumerate(res['behaviorOrder']):
#    plt.plot(moving['Neurons']['Time'], res['behavior'][:,i]+ind*4, color='k', label = res['behaviorLabels'][i], alpha=0.35+0.1*ind)
#    plt.xlabel('Time(s)')
#    
#locs, labels = plt.yticks()
#plt.yticks(np.arange(0,len(res['behaviorOrder'])*4,4), res['behaviorLabels'][res['behaviorOrder']])
##plt.legend()
#plt.subplot(325)
## plot the weights for each PC
#
#for li,line in enumerate(orderedWeights):
#    plt.plot(np.abs(line), label = ('weights for PC{}'.format(li+1)), color='C5', alpha=0.25+0.05*li, lw=1)
#plt.ylabel('Weights')
##plt.xlabel('behaviors')
#plt.plot(np.mean(np.abs(orderedWeights), axis=0), color='C5', alpha=1, lw=2, marker = 'o')
#plt.xticks(np.arange(len(res['behaviorOrder'])), res['behaviorLabels'][res['behaviorOrder']], rotation=30)
#plt.subplot(326)
#
#plt.plot(res['expVariance'], color='C7', alpha=1, lw=2, marker = 'o')
#plt.xticks(np.arange(len(res['behaviorOrder'])),res['behaviorLabels'][res['behaviorOrder']], rotation=30)
#plt.show()