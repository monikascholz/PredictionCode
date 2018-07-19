
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


fig = plt.figure('Fig - 1 : Neural dynamics in freely moving animals', figsize=(9.5, 9*3/4.))
# this gridspec makes one example plot of a heatmap with its PCA
#gs1 = gridspec.GridSpec(4, 3, width_ratios = [1,1,1], height_ratios=[0.1, 1,1,2])
#gsHeatmap = gridspec.GridSpecFromSubplotSpec(4,5, subplot_spec=gs1[0:4,:], width_ratios=[1.25, 0.1, 0.5,0.5,0.5], height_ratios = [0.1,10,10,10], wspace=0.3, hspace=0.25)
gsHeatmap = gridspec.GridSpec(5,4,  width_ratios=[1.5, 0.1, 0.5, 0.5], height_ratios = [1,0.1,0.75,0.1, 0.75])
gsHeatmap.update(left=0.07, right=0.98,  bottom = 0.1, top=0.98, hspace=0.25, wspace=0.45)

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
#first row
#
################################################
# select a special dataset - transiently immobilized
transientData = 'BrainScanner20180511_134913'
transient = data['AML32_chip']['input']['BrainScanner20180511_134913']
transientAnalysis = data['AML32_chip']['analysis']['BrainScanner20180511_134913']
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
         
#heatmap axes
axhm = plt.subplot(gsHeatmap[0,0])
axcb = plt.subplot(gsHeatmap[0,1])
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
axetho = plt.subplot(gsHeatmap[1,0], clip_on=False)
plotEthogram(axetho, time, transient['Behavior']['EthogramFull'], alpha = 1, yValMax=1, yValMin=0, legend=0)
cleanAxes(axetho, 'all')
moveAxes(axetho, 'scaley', 0.02)
axetho.xaxis.label.set_visible(False)
# legend for ethogram
axEthoLeg = plt.subplot(gsHeatmap[1:2,1])#,clip_on=False)
#moveAxes(axEthoLeg, 'right', 0.045)
cleanAxes(axEthoLeg, where='all')
handles, labels = axetho.get_legend_handles_labels()
leg = mpl.legend.Legend(axEthoLeg, handles[::-1], labels[::-1],frameon=False, loc=1, markerscale=0)#,bbox_to_anchor=(-1, 0.9), loc=9)
for hndl in leg.legendHandles:
    hndl._sizes = [0]
axEthoLeg.add_artist(leg);

# plot mean autocorrelation moving versus immobile
movExp = ['AML32_moving', 'AML70_chip']
imExp = ['AML32_immobilized']#, 'AML70_immobilized']

# all moving gfp
movCtrl = ['AML18_moving', 'AML175_moving']
imCtrl = ['AML18_immobilized']

gsPer= gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gsHeatmap[0,2:], wspace=0.1)
colorsExp = {'moving': R1, 'immobilized': B1}
colorCtrl = {'moving': N0,'immobilized': N1}
ax13 = plt.subplot(gsPer[0])
ax14 = plt.subplot(gsPer[1])
ax15 = plt.subplot(gsPer[2])
ax16 = plt.subplot(gsPer[3])
for typ, colors, axes in zip([[movExp, imExp], [movCtrl, imCtrl]], [colorsExp, colorCtrl], [[ax13, ax14], [ax15,ax16]]):
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
        ax.plot(T,tmpdata[0] ,'-',color = colors[condition], lw=2, alpha=1,label = '{} {}'.format(typ, condition))
        #ax.plot(T,np.mean(tmpdata,axis=0) ,'-',color = colors[condition], lw=5, alpha=0.5,label = '{} {}'.format(typ, condition))
        #ax.fill_between(dset[idn]['Period']['Periods'], m-s, m+s, alpha=0.5, zorder=-1,color = colors[condition])
        ax.axhline(color='k', linestyle = '--', zorder=-1)
        ax.set_ylim([-0.2,1])
        ax.text(0.5, 0.9,condition, transform=ax.transAxes, horizontalalignment='center')
#ax13.set_ylabel('Autocorrelation')
ax13.text(-0.5,0,'Autocorrelation', fontsize=14,transform = ax13.transAxes, rotation=90, verticalalignment ='center')
ax15.set_xlabel('Lag (s)')
ax16.set_xlabel('Lag (s)')
ax13.set_xticks([])
ax14.set_xticks([])
ax14.set_yticks([])
ax16.set_yticks([])

# plot autocorr of our favorite dataset
#ax13.plot(transientAnalysis['Period1Half']['Periods'],np.mean(transientAnalysis['Period1Half']['NeuronACorr'],axis=0))
#ax14.plot(transientAnalysis['Period2Half']['Periods'],np.mean(transientAnalysis['Period2Half']['NeuronACorr'],axis=0))

# plot dimensionality for inactive and active plus together
ax3 = plt.subplot(gsHeatmap[2,2])
nComp = 10#results['nComp']
for y, col, lab, mark in zip([results['expVariance'][:nComp],resultshalf['expVariance'][:nComp], results2half['expVariance'][:nComp]]\
        , ['k', R1, B1], ['Full', 'Moving', 'Paralyzed'], ['o', '^', 's']):
    #ax3.fill_between(np.arange(0.5,nComp+0.5),y*100, step='post', color=col, alpha=0.5)
    ax3.plot(np.arange(1,nComp+1),np.cumsum(y)*100, 'o-',color = col, label = lab, marker=mark) 

ax3.set_ylabel('Variance exp. (%)', labelpad=-5)
ax3.set_yticks([0,25,50,75,100])
ax3.set_xlabel('# of components')
ax3.set_xticks([0,5, 10])

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
# principal components
ax4 =plt.subplot(gsHeatmap[2:3,0])#, sharex=axhm)
# plot PCA components
for i in range(np.min([len(results2half['pcaComponents']), 3])):
    #y = results['pcaComponents'][i]
    y = results2half['fullData'][i]
    # normalize
    y =y -np.min(y)
    y =y/np.max(y)
    ax4.text(-100, np.mean(y)+i*1.05, 'PC{}'.format(i+1), color = 'k')
    ax4.plot(time[transient['Neurons']['valid']], i*1.1+y, label='Component {}'.format(i+1), lw=1, color = 'k')

yloc = ax4.get_ylim()[-1]
## indicate immobilization etc
for label, segment in zip(['moving', 'immobilized'], [train, test]):

    ax4.text(np.mean(timeActual[segment]), 1.02*yloc, label,horizontalalignment='center', color=colorsExp[label])
    ax4.plot([timeActual[segment[0]],timeActual[segment[-1]]], [yloc, yloc], color=colorsExp[label])
    
# labels and such
ax4.set_xlabel('Time (s)')
ax4.set_xlim([np.min(timeActual), np.max(timeActual)])
cleanAxes(ax4, where='y')


# plot manifold! MANIFOOOOOLD!
ax5 = plt.subplot(gsHeatmap[2,3], projection='3d', clip_on = False, zorder=-10)
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
colorBy[:6*60*4] = 1 # first four minutes is m9
multicolor(ax5,x[train],y[train],z[train],colorBy[train],c= transientcmap, threedim = True, etho = False, cg = 1)
multicolor(ax5,x[test],y[test],z[test],colorBy[test],c= transientcmap, threedim = True, etho = False, cg = 1)
ax5.scatter3D(x[::12], y[::12], z[::12], c=colorBy[::12], cmap=transientcmap, s=5)
ax5.view_init(elev=15, azim=35)
ax5.dist = 7
axmin, axmax = -1, 1
ticks = [axmin,0, axmax]

ax5.set_xlim([axmin, axmax])
ax5.set_ylim([axmin/2., axmax])
ax5.set_zlim([axmin, axmax])
#
ax5.tick_params(axis='both', which='major', pad=-10)
ax5.axes.xaxis.set_ticklabels([])
ax5.axes.yaxis.set_ticklabels([])
ax5.axes.zaxis.set_ticklabels([])

# make scalebar
axesNames = [ax5.xaxis, ax5.yaxis, ax5.zaxis]
for tmp, loc in zip(axesNames, [(0,0,0),(1,1,1),(2,2,2)]):
    tmp._axinfo['juggled']=loc

# make a scale bar in 3d
scX, scY, scZ = 0,2,-0.04
names = [r'PC$_1$', 'PC$_2$', 'PC$_3$']
align = ['left', 'left','left']
for i in range(3):
    l = np.zeros(3)
    l[i] = 0.25
    ax5.plot([scX, scX +l[0]], [scY, scY+l[1]], [scZ, scZ+l[2]], color='k')
    l = np.zeros(3)+axmin
    l[i] = axmax+0.05
    if i ==0:
        l[i] = axmax+0.1
    ax5.text(l[0], l[1], l[2], names[i], horizontalalignment=align[i],\
        verticalalignment='center', color = 'k')
#ax5.spines['left'].set_position('zero')
moveAxes(ax5, action='left', step=0.02 )
#moveAxes(ax5, action='up', step=0.02 )
moveAxes(ax5, action='scale', step=0.02 )

# pc axes projection
#sciformat = 1.
#multicolor(axproj,x*sciformat,y*sciformat,None,colorBy,c=transientcmap, threedim = False, etho = False, cg = 1)
##axproj.set_xlabel(r'PC1 ($\times \, 10^{-2}$)', labelpad=0, color=Ls[0])
##axproj.set_ylabel(r'PC2 ($\times \, 10^{-2}$)', labelpad=0, color=Ls[1])
#axproj.set_xlabel(r'PC1', labelpad=0, color=Ls[0])
#axproj.set_ylabel(r'PC2', labelpad=0, color=Ls[1])
##moveAxes(axproj, action='up', step=0.02 )
#
##### move everything up a bit
#for axi in [axetho, ax4]:
#    moveAxes(axi, 'up', 0.04)
#################################################
##
## third row - autocorrelations
##
#################################################
# individual rank order correlations for three examples
gsRank = gridspec.GridSpecFromSubplotSpec(1,6,subplot_spec=gsHeatmap[4,:], width_ratios=[1, 1,1,1,0.2,1])
ax8 = plt.subplot(gsRank[1])
ax9 = plt.subplot(gsRank[2])
ax10 = plt.subplot(gsRank[3])
axcbar2 = plt.subplot(gsRank[4])

movData = 'BrainScanner20170613_134800'
immData = 'BrainScanner20180510_092218'
transientData = 'BrainScanner20180511_134913'
for key, dset, label, ax in zip(['AML32_moving', 'AML32_immobilized', 'AML32_chip'],[movData, immData, transientData], ['moving', 'immobilized', 'transient'], [ax8, ax9, ax10]):
    rankC = data[key]['analysis'][dset]['PCArankCorr']
    cax1 = ax.imshow(rankC, vmin=0, vmax=0.5)
    for i in range(3):
        # find best match, delete that option
        ym, xm = np.unravel_index(np.abs(rankC).argmax(), rankC.shape)
        if np.max(rankC)>0.25:
            c = 'k'
        else:
            c='w'
        ax.text(xm,ym, np.round(np.max(rankC), decimals=2), color=c, horizontalalignment ='center', verticalalignment ='center')
        rankC[ym,:] =0
        rankC[:,xm] =0
#    for index, entry in enumerate(np.ravel(rankC)):
#        loc = np.unravel_index(index, rankC.shape)
#        ax.text(loc[0],loc[1], np.round(entry, decimals=2), horizontalalignment ='center', verticalalignment ='center')
    ax.set_title(label)
# colorbar for rank correlations
cbar = fig.colorbar(cax1, cax=axcbar2, use_gridspec = True)
cbar.set_ticks([0,0.5])
cbar.set_ticklabels(['0', '>0.5'])
cbar.outline.set_visible(False)
moveAxes(axcbar2, 'scaley', -0.08)
axcbar2.set_ylabel(r'Correlation', labelpad = -25)
for ax in [ax8, ax9, ax10, axcbar2]:
    moveAxes(ax, 'down', 0.05)

# rank order of PCA weights
ax11 = plt.subplot(gsRank[5])
boxplot = []
for keys in [movExp, imExp]:
    r2 = []
    for key in keys:
        dset = data[key]['analysis']
        for idn in dset.keys():
            
            rankC = np.abs(dset[idn]['PCArankCorr'])
            tmp = []
            for i in range(3):
                # find best match, delete that option
                tmp.append(np.max(rankC))
                xm, ym = np.unravel_index(rankC.argmax(), rankC.shape)
                rankC[xm,:] =0
                rankC[:,ym] =0
            
            r2.append( np.mean(tmp))
            
    boxplot.append(r2)
mkStyledBoxplot(ax11,[1,2], np.array(boxplot), [R1, B1], ['moving', 'immobilized'])
ax11.set_ylim([0,0.5])
ax11.set_yticks([0,0.5])
ax11.set_xlim([0.5,2.5])
# add single star for the transition dataset
ax11.scatter([1.5],np.max(np.abs(transientAnalysis['PCArankCorr'])),s = 15, marker='s',lw=1, color= R1)
ax11.scatter([1.5],np.max(np.abs(transientAnalysis['PCArankCorr'])),s = 15, marker='<',lw=1, color= B1)

plt.show()
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