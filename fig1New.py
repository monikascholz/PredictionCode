
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.ticker as mtick

#
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
gsHeatmap = gridspec.GridSpec(4,5,  width_ratios=[1.5, 0.1, 0.5,0.5,0.5], height_ratios = [1,0.1,1,1], wspace=0.35, hspace=0.25)
gsHeatmap.update(left=0.07, right=0.98,  bottom = 0.07, top=0.98, hspace=0.35, wspace=0.35)

################################################
#
# first row
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
transient = data['AML32_chip']['input']['BrainScanner20180511_134913']
transientAnalysis = data['AML32_chip']['analysis']['BrainScanner20180511_134913']
# time first half, second half. Indices of times
timeHalf = np.arange(0, 1400)
time2Half = np.arange(1600, transient['Neurons']['Activity'].shape[1])
# pull out repeated stuff
time = transient['Neurons']['TimeFull']
timeActual = transient['Neurons']['Time']
noNeurons = transient['Neurons']['Activity'].shape[0]
results = transientAnalysis['PCAHalf2']
resultshalf = transientAnalysis['PCAHalf1']
results2half = transientAnalysis['PCAHalf2']
# plot heatmap ordered by PCA
# colorbar in a nested gridspec because its much better          

#ethogram
axetho = plt.subplot(gsHeatmap[1,0])
# legend for ethogram
axEthoLeg = plt.subplot(gsHeatmap[1,1:3],clip_on=False)
# make bars in correct colors
for i in [-1,0]:
    xmin, xmax = 0, 0.1
    ymin = (i+1)*0.95
    ymax= ymin+0.25
    axEthoLeg.axhspan(ymin, ymax, xmin, xmax, color=colDict[i])
    axEthoLeg.text(xmax, np.mean([ymin, ymax]), labelDict[i], color=colDict[i], verticalalignment ='center')
for i in [1,2]:
    xmin, xmax = 0.5, 0.6
    ymin = (i-1)*0.95
    ymax= ymin+0.25
    axEthoLeg.axhspan(ymin, ymax, xmin, xmax, color=colDict[i])
    axEthoLeg.text(xmax, np.mean([ymin, ymax]), labelDict[i], color=colDict[i], verticalalignment ='center', fontsize=12)
axEthoLeg.spines['left'].set_visible(False)
axEthoLeg.spines['bottom'].set_visible(False)
axEthoLeg.set_yticks([])
axEthoLeg.set_xticks([])

#moveAxes(axEthoLeg, 'scaley', 0.08)
moveAxes(axEthoLeg, 'left', 0.04)
moveAxes(axetho, action='scaley', step=0.01 )
moveAxes(axetho, action='down', step=0.04 )
#
axhm = plt.subplot(gsHeatmap[0,0])
axcb = plt.subplot(gsHeatmap[0,1])

#gsweights = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec=gsHeatmap[1:3,2:], width_ratios=[0.5,1,1], wspace=0.5, hspace=0.5)
# weights
ax2 =plt.subplot(gsHeatmap[0,2])
# weight rank correlation
axweights = plt.subplot(gsHeatmap[0,3])
moveAxes(axweights, 'left', 0.02)
# principal components projected
axproj = plt.subplot(gsHeatmap[0,4])

# principal components
ax4 =plt.subplot(gsHeatmap[2,0], sharex=axhm)
 
 
plotEthogram(axetho, time, transient['Behavior']['EthogramFull'], alpha = 1, yValMax=1, yValMin=0, legend=0)
axetho.set_xticks([])
axetho.xaxis.label.set_visible(False)
moveAxes(axetho, 'scaley', 0.025)
#axetho.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
cax1 = plotHeatmap(time, transient['Neurons']['RawActivity'][results['neuronOrderPCA']], ax=axhm, vmin=-0.5, vmax=2)
axhm.xaxis.label.set_visible(False)
axetho.xaxis.label.set_visible(False)
cbar = fig.colorbar(cax1, cax=axcb, use_gridspec = True)
cbar.set_ticks([-0.5,0,2])
cbar.set_ticklabels(['<-0.5',0,'>2'])
cbar.outline.set_visible(False)
pos = axcb.get_position().get_points()
pos[:,0] -=0.04
posNew = mpl.transforms.Bbox(pos)
axcb.set_position(posNew)
#axcb.set_position()
axcb.set_ylabel(r'$\Delta R/R_0$', labelpad = -25)

# plot the weights
pcs = results['neuronWeights']
# normalize by max for each group
rank = np.arange(0, len(pcs))
print rank
for i in range(np.min([3,pcs.shape[1]])):
    print i
    y= pcs[:,i]
    print len(y), len(results['neuronOrderPCA'])
    ax2.fill_betweenx(rank, np.zeros(noNeurons),y[results['neuronOrderPCA']], step='pre',\
    alpha=1.0-i*0.25, color=Ls[i])
    
#ax2.set_ylabel('Neuron weights')
ax2.set_ylim([0, len(pcs)])
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_yticks([])
ax2.set_xticks([])
moveAxes(ax2, action='left', step=0.03 )

# plot rank of neuron in first vs second half
for pc in range(3):
    for pc2 in range(3):
        rankHalf1, rankHalf2 = np.argsort(resultshalf['neuronWeights'][:,pc]),  np.argsort(results2half['neuronWeights'][:,pc2])
        axweights.scatter(rankHalf1, rankHalf2, alpha=0.75, s = 5,color = Ls[pc] )
        print 'R2', np.corrcoef(rankHalf1, rankHalf2)[0,1]
axweights.set_xlabel('rank moving')
axweights.set_ylabel('rank immobilized')

# plot dimensionality for inactive and active plus together
#nComp = 10#results['nComp']
#for y, col, lab in zip([results['expVariance'][:nComp],results2half['expVariance'][:nComp], resultshalf['expVariance'][:nComp]]\
#        , ['k', R1, B1], ['Full', 'Moving', 'Paralyzed']):
#    #ax3.fill_between(np.arange(0.5,nComp+0.5),y*100, step='post', color=col, alpha=0.5)
#    ax3.plot(np.arange(1,nComp+1),np.cumsum(y)*100, 'o-',color = col, lw=1, markersize =3, label = lab) 
#ax3.legend(loc=4, fontsize=12)
#ax3.set_ylabel('Explained variance (%)', labelpad=-5)
#ax3.set_yticks([0,25,50,75,100])
#ax3.set_xlabel('# of components')
#ax3.set_xticks([0,5, 10])

#### move everything up a bit
#for axi in [axhm, ax2, axweights, axcb,axproj ]:
#    moveAxes(axi, 'up', 0.04)


#################################################
##
## second row
##
#################################################
#
# plot PCA components
for i in range(np.min([len(results['pcaComponents']), 3])):
    #y = results['pcaComponents'][i]
    y = results['fullData'][i]
    # normalize
    y =y -np.min(y)
    y =y/np.max(y)
    ax4.text(-100, np.mean(y)+i*1.05, 'PC{}'.format(i+1), color = Ls[i])
    ax4.plot(time[transient['Neurons']['valid']], i*1.1+y, label='Component {}'.format(i+1), lw=1, color = Ls[i])
## draw a box for the testset
#ax4.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N2, zorder=-10, alpha=0.75)
#ax4.text(np.mean(timeActual[test]), ax4.get_ylim()[-1], 'Testset',horizontalalignment='center')
ax4.set_xlabel('Time (s)')
ax4.set_xlim([np.min(timeActual), np.max(timeActual)])
ax4.spines['left'].set_visible(False)
ax4.set_yticks([])
moveAxes(ax4, action='up', step=0.02 )
# plot manifold! MANIFOOOOOLD!
ax5 = plt.subplot(gsHeatmap[2,2:5], projection='3d')
# plot manifold for split dataset
x,y,z = results['fullData'][:3]
# make smoooth
smooth = 12
x = gaussian_filter1d(x, smooth)
y = gaussian_filter1d(y, smooth)
z = gaussian_filter1d(z, smooth)
# color by before and after
colorBy = np.zeros(len(timeActual))
colorBy[:6*60*4] = 1 # first four minutes is m9
multicolor(ax5,x,y,z,colorBy,c= transientcmap, threedim = True, etho = False, cg = 1)
ax5.scatter3D(x[::12], y[::12], z[::12], c=colorBy[::12], cmap=transientcmap, s=10)
ax5.view_init(elev=40, azim=-15)
ax5.dist = 7
axmin, axmax = -5, 5
ticks = [axmin,0, axmax]

ax5.set_xlim([axmin, axmax])
ax5.set_ylim([axmin, axmax])
ax5.set_zlim([axmin, axmax])
#
ax5.tick_params(axis='both', which='major', pad=0)
ax5.axes.xaxis.set_ticklabels([])
ax5.axes.yaxis.set_ticklabels([])
ax5.axes.zaxis.set_ticklabels([])

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
    l[i] = 1
    ax5.plot([scX, scX +l[0]], [scY, scY+l[1]], [scZ, scZ+l[2]], color='k')
    l = np.zeros(3)+axmin
    l[i] = axmax+0.0075*2.5
    if i ==2:
        l[i] = axmax+0.0075
    ax5.text(l[0], -l[1], l[2], names[i], horizontalalignment=align[i],\
        verticalalignment='center', color = Ls[i])

moveAxes(ax5, action='left', step=0.04 )
#moveAxes(ax5, action='down', step=0.04 )
#moveAxes(ax5, action='scaley', step=0.02 )

# pc axes projection
sciformat = 1.
multicolor(axproj,x*sciformat,y*sciformat,None,colorBy,c=transientcmap, threedim = False, etho = False, cg = 1)
#axproj.set_xlabel(r'PC1 ($\times \, 10^{-2}$)', labelpad=0, color=Ls[0])
#axproj.set_ylabel(r'PC2 ($\times \, 10^{-2}$)', labelpad=0, color=Ls[1])
axproj.set_xlabel(r'PC1', labelpad=0, color=Ls[0])
axproj.set_ylabel(r'PC2', labelpad=0, color=Ls[1])
#moveAxes(axproj, action='up', step=0.02 )

#### move everything up a bit
for axi in [axetho, ax4]:
    moveAxes(axi, 'up', 0.04)
#################################################
##
## third row - autocorrelations
##
#################################################
# plot mean autocorrelation moving versus immobile
# Todo variance explained for moving and immobile
colorsExp = {'moving': R1, 'immobilized': B1}
colorsCtrl = {'moving': N0,'immobilized': N1}

# all moving gcamps
movExp = ['AML32_moving', 'AML70_chip']
imExp = ['AML32_immobilized', 'AML70_immobilized']


# all moving gfp
movCtrl = ['AML18_moving', 'AML175_moving']
imCtrl = ['AML18_immobilized']


gsPer= gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gsHeatmap[3,:], wspace=0.1)
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
                results=  dset[idn]['Period']
                
                tmpdata.append(np.mean(dset[idn]['Period']['NeuronACorr'],axis=0)) 
                T = dset[idn]['Period']['Periods']
        m, s = np.nanmean(tmpdata, axis=0), np.nanstd(tmpdata, axis=0)
        tmpdata= np.array(tmpdata)
        ax.plot(T,tmpdata.T ,'-',color = colors[condition], lw=1, alpha=0.35,label = '{} {}'.format(typ, condition))
        ax.plot(T,tmpdata[0] ,'-',color = colors[condition], lw=2, alpha=1,label = '{} {}'.format(typ, condition))
        #ax.plot(T,np.mean(tmpdata,axis=0) ,'-',color = colors[condition], lw=5, alpha=0.5,label = '{} {}'.format(typ, condition))
        #ax.fill_between(dset[idn]['Period']['Periods'], m-s, m+s, alpha=0.5, zorder=-1,color = colors[condition])
        ax.axhline(color='k', linestyle = '--', zorder=-1)
        ax.set_ylim([-0.2,1])
        

        ax.set_xlabel('Lag (s)')
ax13.set_ylabel('Autocorrelation')
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