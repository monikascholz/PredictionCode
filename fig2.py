
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
#
colorsExp = {'moving': R1, 'immobilized': B1}
colorCtrl = {'moving': N0,'immobilized': N1}
################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML32', 'AML18']:
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


fig = plt.figure('Fig - 2 : Behavior is represented in the brain', figsize=(9.5, 9))
# this gridspec makes one example plot of a heatmap with its PCA
gs1 = gridspec.GridSpec(4, 3, width_ratios = [1,0.5,0.5])
gs1.update(left=0.07, right=0.98, wspace=0.25, bottom = 0.07, top=0.97, hspace=0.25)

################################################
#
# first row
#
################################################
#ax2 = plt.subplot(gs1[0,1])
#ax3 = plt.subplot(gs1[0,2])

# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B', 'C']
y0 = 0.97
locations = [(0,y0),  (0.55,y0), (0.76,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
            horizontalalignment='left',verticalalignment='top',)
################################################
#
#first row
#
################################################
# select a special dataset - moving AML32
movingAML32 = 'BrainScanner20170613_134800'
moving = data['AML32_moving']['input'][movingAML32]
movingAnalysis = data['AML32_moving']['analysis'][movingAML32]
label = 'AngleVelocity'
splits = movingAnalysis['Training']
train, test = splits[label]['Train'], splits[label]['Test']
# pull out repeated stuff
time = moving['Neurons']['TimeFull']
timeActual = moving['Neurons']['Time']
noNeurons = moving['Neurons']['Activity'].shape[0]
results = movingAnalysis['PCA']
# plot heatmap ordered by PCA
# colorbar in a nested gridspec because its much better          
gsHeatmap = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs1[0:3,:], width_ratios=[0.9,0.05,0.5,0.3], height_ratios = [0.25,5,5, 10], wspace=0.3, hspace=0.5)
axhm = plt.subplot(gsHeatmap[1,0])
axcb = plt.subplot(gsHeatmap[1,1])
axetho = plt.subplot(gsHeatmap[0,0])
pos = axetho.get_position().get_points()
pos[0,1] -=0.01
pos[1,1] +=0.01
pos[:,1] -=0.01
posNew = mpl.transforms.Bbox(pos)
axetho.set_position(posNew)
ax2 =plt.subplot(gsHeatmap[1,2])
ax3 =plt.subplot(gsHeatmap[1,3])
ax4 =plt.subplot(gsHeatmap[2,0])

mp.plotEthogram(axetho, time, moving['Behavior']['EthogramFull'], alpha = 1, yValMax=1, yValMin=0, legend=0)
axetho.set_xticks([])
axetho.xaxis.label.set_visible(False)
#axetho.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
cax1 = mp.plotHeatmap(time, moving['Neurons']['ActivityFull'][results['neuronOrderPCA']], ax=axhm, vmin=-0.5, vmax=1)
axhm.xaxis.label.set_visible(False)
axetho.xaxis.label.set_visible(False)
cbar = fig.colorbar(cax1, cax=axcb, use_gridspec = True)
cbar.set_ticks([-0.5,0,1])
cbar.set_ticklabels(['<-0.5',0,'>1'])
cbar.outline.set_visible(False)
pos = axcb.get_position().get_points()
pos[:,0] -=0.02
posNew = mpl.transforms.Bbox(pos)
axcb.set_position(posNew)
#axcb.set_position()
axcb.set_ylabel(r'$\Delta R/R_0$', labelpad = 0)

# plot the weights
pcs = movingAnalysis['PCA']['neuronWeights']
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

# plot dimensionality for inactive and active plus together
nComp = 10#results['nComp']
for y, col in zip([results['expVariance'][:nComp]], ['k']):
    ax3.fill_between(np.arange(0.5,nComp+0.5),y*100, step='post', color=col, alpha=0.5)
    ax3.plot(np.arange(1,nComp+1),np.cumsum(y)*100, 'o-',color = col, lw=1, markersize =3) 

ax3.set_ylabel('Variance explained (%)')
ax3.set_yticks([0,25,50,75,100])
ax3.set_xlabel('PCA components')
ax3.set_xticks([0,5, 10])

################################################
#
# second row
#
################################################
#gsPCs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gsHeatmap[2:3,1:], width_ratios=[2,1],height_ratios=[2], wspace=0.35, hspace=0.2)

# plot PCA components
for i in range(np.min([len(results['pcaComponents']), 3])):
    y = results['pcaComponents'][i]
    # normalize
    y =y -np.min(y)
    y =y/np.max(y)
    ax4.text(-100, np.max(y)+i*1.1, 'Component {}'.format(i+1), color = Ls[i])
    ax4.plot(time[moving['Neurons']['valid']], i*1.1+y, label='Component {}'.format(i+1), lw=1, color = Ls[i])
# draw a box for the testset
ax4.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N2, zorder=-10, alpha=0.75)
ax4.text(np.mean(timeActual[test]), ax4.get_ylim()[-1], 'Testset',horizontalalignment='center')
ax4.set_xlabel('Time (s)')
ax4.set_xlim([np.min(timeActual), np.max(timeActual)])
ax4.spines['left'].set_visible(False)
ax4.set_yticks([])


ax5 = plt.subplot(gsHeatmap[2,1:3], projection='3d')
# color PCA by ethogram
x,y,z = movingAnalysis['PCA']['pcaComponents'][:3]
# make smoooth
smooth = 12
x = gaussian_filter1d(x, smooth)
y = gaussian_filter1d(y, smooth)
z = gaussian_filter1d(z, smooth)
# color by ethogram
colorBy = np.reshape(np.array(moving['Behavior']['Ethogram']), (-1, ))
mp.multicolor(ax5,x,y,z,colorBy,c= mp.ethocmap, threedim = True, etho = True, cg = 1)
ax5.scatter3D(x[::12], y[::12], z[::12], c=colorBy[::12], cmap=mp.ethocmap, s=10)
ax5.view_init(elev=20, azim=70)
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
ax5.axes.xaxis.set_ticklabels([])
ax5.axes.yaxis.set_ticklabels([])
ax5.axes.zaxis.set_ticklabels([])
#ax5.tick_params(axis='both', which='major', pad=-1)
p = 0
#ax5.set_xlabel('\nPC1', fontsize=10,labelpad =p,)
#ax5.set_ylabel('\nPC2', fontsize=10,labelpad =p,)
#ax5.set_zlabel('\nPC3', labelpad =p, fontsize=10)
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
    l[i] = axmax+0.005
    ax5.text(l[0], l[1], l[2], names[i], horizontalalignment=align[i],\
        verticalalignment='center')
pos = ax5.get_position().get_points()
pos[:,0] -=0.02
posNew = mpl.transforms.Bbox(pos)
ax5.set_position(posNew)
    #ax5.text(scX +l[0]*0.5,scY+l[1]*0.5,scZ+l[2]*0.5,names[i], color='k')
#ax5.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
# pc axes projection
axproj = plt.subplot(gsHeatmap[2,3])
mp.multicolor(axproj,x,y,None,colorBy,c=mp.ethocmap, threedim = False, etho = True, cg = 1)
axproj.set_xlabel('PC1', labelpad=0)
axproj.set_ylabel('PC2', labelpad=-10)
axproj.set_xticks([-0.04,0, 0.04])
axproj.set_yticks([-0.04, 0, 0.04])

plt.setp(axproj.get_xticklabels(), rotation=-25)
axproj.ticklabel_format(style='sci',scilimits=(0,0),axis='both')

################################################
#
# third row
#
################################################
gsEWs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gsHeatmap[3,0], wspace=0.1, hspace=0.4, height_ratios=[2,1,1])
# plot eigenworm schematic
ax6 = plt.subplot(gsEWs[0,0],  adjustable='box')
ax6.axis('equal')
# plot eigenworms
eigenworms = dh.loadEigenBasis(filename = 'utility/Eigenworms.dat', nComp=3, new=True)
lengths = np.array([4])
meanAngle = np.array([-0.07945])
refPoint = np.array([[0,0]])
prefactor = [r'',r'$=a_1$ *',r'+ $a_2$ *',r'+ $a_3$ *' ][::-1]
descr = ['posture', 'undulation', 'undulation', 'turn'][::-1]
colors = [B1, B1, R1]
for i in range(4):
    pcs = np.zeros(3)
    if i ==3:
        pcsNew, meanAngle, _, refPoint = dh.calculateEigenwormsFromCL(moving['CL'], eigenworms)
        pc3New, pc2New, pc1New = pcsNew
        cl = dh.calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint)
        
        new = mp.createWorm(cl[0,:,0], cl[0,:,1])
    else:
        pcs[i] = 10
        clNew = dh.calculateCLfromEW(pcs, eigenworms[:3], meanAngle, lengths, refPoint)[0]
        new = mp.createWorm(clNew[:,0], clNew[:,1])
    
    new -=np.mean(new, axis=0)
    new[:,0] -= i*600 - 800
    x, y = np.mean(new, axis =0)
    p2 = mpl.patches.Polygon(new, closed=True, fc=N0, ec='none')
    plt.text(x,y-200, descr[i], fontsize=12, horizontalalignment = 'center')
    plt.text(x-250,y, prefactor[i], fontsize=12, horizontalalignment = 'center')
    ax6.add_patch(p2)
    ax6.axes.get_xaxis().set_visible(False)
    ax6.axes.get_yaxis().set_visible(False)
    ax6.spines['left'].set_visible(False)
    ax6.spines['bottom'].set_visible(False)
    ax6.set_xlim(-1200, 1000)
    ax6.set_ylim(-50, 10)


# plot angle velocity and turns
ax7 = plt.subplot(gsEWs[1,0])
ax7.plot(moving['Neurons']['Time'], moving['Behavior']['AngleVelocity'], color = R1)
ax7.set_ylabel('Wave speed')
#ax7.axes.get_xaxis().set_visible(False)
#ax7.spines['bottom'].set_visible(False)
ax7.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
ax8 = plt.subplot(gsEWs[2,0])
ax8.plot(moving['Neurons']['Time'],moving['Behavior']['Eigenworm3'], color = B1)
ax8.set_ylabel('Turn')
ax8.set_xlabel('Time (s)')

# schematic of reverse prediction
gsScheme = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gsHeatmap[3,1:], wspace=0.1, hspace=0.4)
axscheme1 =plt.subplot(gsScheme[0,0])
axscheme1.axis('equal')
# input are behaviors
#
x = moving['Neurons']['Time'][test]
circles = []
for linex, label in enumerate(movingAnalysis['RevPred']['behaviorLabels'][movingAnalysis['RevPred']['behaviorOrder']]):
    axscheme1.text(0, linex*10, label, horizontalalignment='right')
    circ = mpl.patches.Circle((10, linex*10), 2.5,fill=False, ec=R1)
    axscheme1.text(10, linex*10, r'$w_{}$'.format(linex+1))
    circles.append(circ)
    axscheme1.plot([12.5,20],[linex*10, 25], color='k')
    # TODO add behavior lines
    #plt.plot(x, movingAnalysis['RevPred']['behavior'][:,i]+linex*4, color='k')
    
collection = mpl.collections.PatchCollection(circles, color='none', edgecolor=R1)
axscheme1.add_collection(collection)

axscheme2 =plt.subplot(gsScheme[0,1])
res = movingAnalysis['RevPred']
newHM = res['predictedNeuralDynamics']
orderedWeights = res['behaviorWeights'][:,res['behaviorOrder']]
gsPred = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs1[3,0], wspace=0.1, hspace=0.4, height_ratios=[1])
#

for ind, i in enumerate(res['PCA_indices'][:4]):
    
    line1, = axscheme2.plot(x, res['NeuralPCS'][test,i]+ind*12, color='k', label='Neural PCs')
    line2, = axscheme2.plot(x, res['predictedNeuralPCS'][test,i]+ind*12, color='C3', label= 'Predicted')
    axscheme2.text(x[-1]*0.9, 1.2*np.max(res['predictedNeuralPCS'][test,i]+ind*10), '$R^2={:.2f}$'.format(res['R2_test'][ind]))
#plt.legend([line1, line2], ['Neural PCs', 'Predicted from Behavior'], loc=2)
ylabels = ['PC {}'.format(index+1) for index in res['PCA_indices'][:4]]
axscheme2.set_yticks(np.arange(0,4*12, 12))
axscheme2.set_yticklabels(ylabels)
axscheme2.set_xlabel('Time(s)')
################################################
#
# fourth row
#
################################################
# predict neural activity from behavior
#



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
plt.figure('PredictedNeuralActivity', figsize=(2.28*4,2.28*6))

# show reduced dimensionality heatmap
mp.plotHeatmap(moving['Neurons']['Time'][test], res['lowDimNeuro'][:,test])
plt.subplot(322)
mp.plotHeatmap(moving['Neurons']['Time'][test], newHM[:,test], vmin=np.min(newHM)*1.1, vmax=np.max(newHM)*0.9)
plt.subplot(324)
for ind, i in enumerate(res['PCA_indices'][:4]):
    x = moving['Neurons']['Time'][test]
    line1, = plt.plot(x, res['NeuralPCS'][test,i]+ind*12, color='C0', label='Neural PCs')
    line2, = plt.plot(x, res['predictedNeuralPCS'][test,i]+ind*12, color='C3', label= 'Predicted')
    plt.text(x[-1]*0.9, 1.2*np.max(res['predictedNeuralPCS'][test,i]+ind*10), '$R^2={:.2f}$'.format(res['R2_test'][ind]))
plt.legend([line1, line2], ['Neural PCs', 'Predicted from Behavior'], loc=2)
ylabels = ['PC {}'.format(index+1) for index in res['PCA_indices'][:4]]
plt.yticks(np.arange(0,4*12, 12), ylabels)
plt.xlabel('Time(s)')
plt.subplot(323)
for ind, i in enumerate(res['behaviorOrder']):
    plt.plot(moving['Neurons']['Time'], res['behavior'][:,i]+ind*4, color='k', label = res['behaviorLabels'][i], alpha=0.35+0.1*ind)
    plt.xlabel('Time(s)')
    
locs, labels = plt.yticks()
plt.yticks(np.arange(0,len(res['behaviorOrder'])*4,4), res['behaviorLabels'][res['behaviorOrder']])
#plt.legend()
plt.subplot(325)
# plot the weights for each PC

for li,line in enumerate(orderedWeights):
    plt.plot(np.abs(line), label = ('weights for PC{}'.format(li+1)), color='C5', alpha=0.25+0.05*li, lw=1)
plt.ylabel('Weights')
#plt.xlabel('behaviors')
plt.plot(np.mean(np.abs(orderedWeights), axis=0), color='C5', alpha=1, lw=2, marker = 'o')
plt.xticks(np.arange(len(res['behaviorOrder'])), res['behaviorLabels'][res['behaviorOrder']], rotation=30)
plt.subplot(326)

plt.plot(res['expVariance'], color='C7', alpha=1, lw=2, marker = 'o')
plt.xticks(np.arange(len(res['behaviorOrder'])),res['behaviorLabels'][res['behaviorOrder']], rotation=30)
plt.show()