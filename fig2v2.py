
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:15:14 2018
Figure 2 - Behavior is represented in the brain
@author: monika
"""
import numpy as np
import matplotlib as mpl
#
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage.filters import gaussian_filter1d
#
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
for typ in ['AML32', 'AML18', 'AML175', 'AML70']:
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


fig = plt.figure('Fig - 2 : Predicting behavior from neural dynamics', figsize=(9.5,3*2.25))
# this gridspec makes one example plot of a heatmap with its PCA
gs1 = gridspec.GridSpec(4, 4, width_ratios = [1,0.1,0.5,0.5], height_ratios=[1,0.1,0.9,1.25])
gs1.update(left=0.07, right=0.98,  bottom = 0.1, top=0.98, hspace=0.25, wspace=0.25)
################################################
#
# letters
#
################################################

################################################
# mark locations on the figure to get good guess for a,b,c locs
#for y in np.arange(0,1.1,0.1):
#    plt.figtext(0, y, y)
#for x in np.arange(0,1.1,0.1):
#    plt.figtext(x, 0.95, x)
#
#letters = map(chr, range(65, 91)) 
## add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B', 'C', 'D']
x0 = 0
locations = [(x0,0.97),  (x0,0.69), (x0,0.6),  (x0,0.39)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
#
letters = ['E','F']
y0 = 0.97
locations = [(0.54,y0), (0.785,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
#            horizontalalignment='left',verticalalignment='top',)
letters = ['G','H']
y0 = 0.69
locations = [(0.54,y0), (0.785,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
letters = ['I']
y0 = 0.39
locations = [(0.49,y0), (0.77,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
#            horizontalalignment='left',verticalalignment='top',)
################################################
#
#first row
#
################################################
# select a special dataset - moving AML32
movingAML32 = 'BrainScanner20170613_134800'#'BrainScanner20170424_105620'#'
moving = data['AML32_moving']['input'][movingAML32]
movingAnalysis = data['AML32_moving']['analysis'][movingAML32]
label = 'AngleVelocity'
splits = movingAnalysis['Training']
train, test = splits[label]['Train'], splits[label]['Test']
# pull out repeated stuff
time = moving['Neurons']['TimeFull']
timeActual = moving['Neurons']['Time']
t = moving['Neurons']['Time'][test]
noNeurons = moving['Neurons']['Activity'].shape[0]
results = movingAnalysis['PCA']

# plot heatmap ordered by PCA
# colorbar in a nested gridspec because its much better          
#heatmap axes
axhm = plt.subplot(gs1[0,0])
axcb = plt.subplot(gs1[0,1])
axetho = plt.subplot(gs1[1,0], clip_on=False)
axEthoLeg = plt.subplot(gs1[1:2,1])#,clip_on=False)
#heatmap
cax1 = plotHeatmap(time, moving['Neurons']['ActivityFull'][results['neuronOrderPCA']], ax=axhm, vmin=-0.5, vmax=2)
axhm.xaxis.label.set_visible(False)
axhm.set_xticks([])
# colorbar
cbar = fig.colorbar(cax1, cax=axcb, use_gridspec = True)
cbar.set_ticks([-0.5,0,2])
cbar.set_ticklabels(['<-0.5',0,'>2'])
cbar.outline.set_visible(False)
moveAxes(axcb, 'left', 0.04)
moveAxes(axcb, 'scaley', -0.08)
moveAxes(axcb, 'scalex', -0.02)
axcb.set_ylabel(r'$\Delta R/R_0$', labelpad = -25)
#ethogram

plotEthogram(axetho, time, moving['Behavior']['EthogramFull'], alpha = 1, yValMax=1, yValMin=0, legend=0)
cleanAxes(axetho, 'all')
moveAxes(axetho, 'scaley', 0.02)
axetho.xaxis.label.set_visible(False)
# legend for ethogram

moveAxes(axEthoLeg, 'right', 0.02)
moveAxes(axEthoLeg, 'up', 0.025)
cleanAxes(axEthoLeg, where='all')
handles, labels = axetho.get_legend_handles_labels()
handles, labels = axetho.get_legend_handles_labels()
leg = mpl.legend.Legend(axEthoLeg, handles[::-1], labels[::-1],frameon=1, loc=1,prop={'size':12},handlelength=0.5, labelspacing=0,handletextpad=0.5)#,bbox_to_anchor=(-1, 0.9), loc=9)
for hndl in leg.legendHandles:
    hndl._sizes = [0]
axEthoLeg.add_artist(leg);

ax4 = plt.subplot(gs1[2,0])
# plot PCA components
for i in range(np.min([len(results['pcaComponents']), 3])):
    y = results['pcaComponents'][i]
    # normalize
    y =y -np.min(y)
    y =y/np.max(y)
    ax4.text(-100, np.mean(y)-i*1.15, 'PC{}'.format(i+1), color = 'k')
    ax4.plot(time[moving['Neurons']['valid']], -i*1.1+y, label='Component {}'.format(i+1), lw=1, color = 'k')
# draw a box for the testset
ax4.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N2, zorder=-10, alpha=0.75)
ax4.text(np.mean(timeActual[test]), ax4.get_ylim()[-1], 'Testset',horizontalalignment='center')
#ax4.set_xlabel('Time (s)')
#ax4.set_xlim([np.min(timeActual), np.max(timeActual)])
#ax4.spines['left'].set_visible(False)
#ax4.set_yticks([])
cleanAxes(ax4)
gsEWs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs1[3,0], wspace=0.1, hspace=0.2, height_ratios=[1,1])

# plot angle velocity and turns
ax7 = plt.subplot(gsEWs[0,0])
ax7.plot(timeActual, moving['Behavior']['AngleVelocity'], color = R1)
# draw a box for the testset
ax7.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N2, zorder=-10, alpha=0.75)
ax7.axhline(color='k', linestyle = '--', zorder=-1)
#ax7.text(np.mean(timeActual[test]), ax4.get_ylim()[-1], 'Testset',horizontalalignment='center')
ax7.set_ylabel('Velocity')
ax7.set_xlim([timeActual[0], timeActual[-1]])
# make scalebar
xscale = timeActual[0]-20
yscale =  [-0.025, 0.025]
#ax7.plot([xscale, xscale], yscale, color=R1, clip_on=False)
#ax7.text(xscale, np.max(ax7.get_ylim())*1.1, 'Velocity', color=R1,horizontalalignment='center',verticalalignment='center')
#ax7.text(xscale, 0, np.ptp(yscale), color=R1, rotation = 90,horizontalalignment='right',verticalalignment='center')
#ax7.axes.get_yaxis().set_visible(False)
#ax7.spines['left'].set_visible(False)
# remove xlabels
plt.setp(ax7.get_xticklabels(), visible=False)
ax8 = plt.subplot(gsEWs[1,0])
ax8.plot(timeActual,moving['Behavior']['Eigenworm3'], color = B1)
# draw a box for the testset
ax8.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N2, zorder=-10, alpha=0.75)
#ax8.text(np.mean(timeActual[test]), ax4.get_ylim()[-1], 'Testset',horizontalalignment='center')
ax8.axhline(color='k', linestyle ='--', zorder=-1)
ax8.set_ylabel('Turn', labelpad=-7)
ax8.get_yaxis().set_label_coords(-0.12, 0.5)
ax7.get_yaxis().set_label_coords(-0.12, 0.5)
ax8.set_xlabel('Time (s)')
ax8.set_xlim([timeActual[0], timeActual[-1]])

################################################
#
# right column
#
################################################
# one nice gridspec
gsPred = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs1[:,2:],height_ratios=[1,1,1.5],width_ratios=[1.5,1], hspace=0.35, wspace=0.5)

# output of behavior prediction from PCA
flag = 'PCAPred'
ybeh = [0, -6]
axscheme1 = plt.subplot(gsPred[0,0])
axscheme1.set_title('PCA model', y=1.05)
for behavior, color, cpred, yl, label, align in zip(['AngleVelocity','Eigenworm3' ], \
            [N1, N1], [R1, B1], ybeh, ['Velocity', 'Turn'], ['center', 'center']):
    beh = moving['Behavior'][behavior]

    meanb, maxb = np.nanmean(beh),np.nanstd(beh)
    beh = (beh[test]-meanb)/maxb
    
    behPred = movingAnalysis[flag][behavior]['output'][test]
    #behPred = (behPred-meanb)/maxb
    
    axscheme1.plot(t, beh+yl, color=color)
    axscheme1.plot(t, behPred+yl, color=cpred)
    axscheme1.text(t[-1], np.max(yl+beh)*1.3, \
    r'$R^2 = {:.2f}$'.format(np.float(movingAnalysis[flag][behavior]['scorepredicted'])), horizontalalignment = 'right')
    axscheme1.text(t[0]*0.8, yl, label, rotation=90, color=cpred, verticalalignment=align)

# add scalebar
#l =120
#y = axscheme1.get_ylim()[0]
#axscheme1.plot([t[0], t[0]+l],[y, y], 'k', lw=2)
#axscheme1.text(t[0]+l*0.5,y*0.95, '2 min', horizontalalignment='center')
cleanAxes(axscheme1)


# output of behavior prediction from elastic net
flag = 'ElasticNet'
ybeh = [0, -6]
axscheme2 = plt.subplot(gsPred[1,0], sharex=axscheme1)
axscheme2.set_title('Sparse linear model', y=1.05)

for behavior, color, cpred, yl, label, align in zip(['AngleVelocity','Eigenworm3' ], \
            [N1, N1], [R1, B1], ybeh, ['Velocity', 'Turn'], ['center', 'center']):
    beh = moving['Behavior'][behavior]

    meanb, maxb = np.nanmean(beh),np.nanstd(beh)
    beh = (beh[test]-meanb)/maxb
    
    behPred = movingAnalysis[flag][behavior]['output'][test]
    behPred = (behPred-meanb)/maxb
    
    axscheme2.plot(t, beh+yl, color=color)
    axscheme2.plot(t, behPred+yl, color=cpred)
    axscheme2.text(t[-1], np.max(yl+behPred), \
    r'$R^2 = {:.2f}$'.format(np.float(movingAnalysis[flag][behavior]['scorepredicted'])), horizontalalignment = 'right')
    axscheme2.text(t[0]*0.8, yl, label, rotation=90, color=cpred, verticalalignment=align)

# add scalebar
l =120
y = axscheme2.get_ylim()[0]
axscheme2.plot([t[0], t[0]+l],[y, y], 'k', lw=2)
axscheme2.text(t[0]+l*0.5,y*0.93, '2 min', horizontalalignment='center')
cleanAxes(axscheme2)
moveAxes(axscheme1, 'down', 0.04)
moveAxes(axscheme2, 'down', 0.04)
###################################
# scatter of Prediction versus True
###################################
gsScatter = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gsPred[1,1])
axscatter1 = plt.subplot(gsScatter[0])
#axscatter2 = plt.subplot(gsScatter[1])

for behavior, cpred, yl, label, ax in zip(['AngleVelocity','Eigenworm3' ], \
            [R1, B1], ybeh, ['Velocity', 'Turn'],[axscatter1, axscatter1] ):
    beh = moving['Behavior'][behavior][test]
    behPred = movingAnalysis[flag][behavior]['output'][test]
    maxb=np.max(beh)
    beh/=maxb
    behPred/=maxb
    xPlot, y, yerr = sortnbin(beh, behPred, nBins=10, rng=[np.min(beh), np.max(beh)])
    
    ax.errorbar(xPlot, y, yerr=yerr, color=cpred, linestyle='None', marker='o')
    
    ax.plot([min(xPlot),max(xPlot)], [min(xPlot), max(xPlot)], 'k:')
    
    #ax.text(0.5, 1, label, transform =ax.transAxes, horizontalalignment='center')
axscatter1.set_ylabel('Predicted')
axscatter1.set_xlabel('True')
########################################
# predict behavior from PCA axes
########################################
flag = 'PCAPred'

axR2 = plt.subplot(gsPred[0,1])
scores = []
locs = [0, 0.75]
for behavior, color, loc in zip(['AngleVelocity', 'Eigenworm3'], [R1, B1], locs):
    alldata = []
    for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
        dset = data[key]['analysis']
        keep = []
        for idn in dset.keys():
            results=  dset[idn][flag][behavior]
            keep.append(results['scorepredicted'])
            
        keep = np.array(keep)
        rnd1 = np.random.rand(len(keep))*0.2
        axR2.scatter(np.zeros(len(keep))+rnd1+loc+0.15, keep, marker = marker,s=15, c = color, alpha=0.5)
        alldata.append(keep)
    scores.append(np.concatenate(alldata))
scores = np.array(scores)
print scores
mkStyledBoxplot(axR2, locs, scores, [R1, B1], ['Velocity', 'Turn'], scatter=False, rotate=False)
axR2.axhline(linestyle=':', color='k')

axR2.set_xlim([-0.25, 1.25])
axR2.set_ylabel(r'$R^2$ (Testset)', labelpad=-10)

################################################################
# Plot test results!
###################################################################
flag='ElasticNet'
#gsLasso = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gsPred[2,:])#, width_ratios=[2,1], height_ratios=[2,1])

# double for broken axis
gsLasso = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsPred[2,:], height_ratios=[3,1])#, width_ratios=[2,1], height_ratios=[2,1])
axV = plt.subplot(gsLasso[0])
axVb = plt.subplot(gsLasso[1],zorder=-10, fc='none')
#axT = plt.subplot(gsLasso[0])
## designate colors
colorsExp = {'AngleVelocity': R1, 'Eigenworm3': B1}
colorCtrl = {'AngleVelocity': N0,'Eigenworm3': N1}
# get all the scores for gcamp
gcamp = []
for behavior, xoff in zip(['AngleVelocity', 'Eigenworm3'], [0, 3]):
    scores = []
    for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
        dset = data[key]['analysis']
        keep = []
        for idn in dset.keys():
            results=  dset[idn][flag][behavior]
            try:
                keep.append(np.array([results['scorepredicted'], np.max(results['individualScore'])]))
            except ValueError:
                keep.append(np.array([results['scorepredicted'], 0]))
        # do some plotting
        keep = np.array(keep)
        rnd1 = np.random.rand(len(keep))*0.2
        rnd2 = np.random.rand(len(keep))*0.2
        c = colorsExp[behavior]
        #axV.set_color_cycle( [c if line[0]>line[1] else N2 for line in keep])
        axV.scatter(xoff+np.zeros(len(keep))+rnd1, keep[:,0], marker = marker,c = c, edgecolor=c, alpha=0.5, s=25)
        axV.scatter(xoff+np.ones(len(keep))+rnd2, keep[:,1], marker = marker, c = c, edgecolor=c, alpha=0.5, s=25)
        
        axV.plot(np.vstack([rnd1, 1+rnd2])+xoff, keep.T, zorder=-2, linestyle=':', color=c)
   
        scores.append(keep)
    gcamp.append(np.concatenate(scores, axis=0))
gcamp = np.array(gcamp)
axV.axhline(color='k', linestyle=':')
# now boxplot for the full set
binloc = [-0.25, 2.25]
x0 = -0.25
mkStyledBoxplot(axV, [x0, x0+3],gcamp[:,:,0],[R1, B1], [1,2],scatter=False, dx=1.25  )
mkStyledBoxplot(axV, [x0+1.75, x0+1.75+3],gcamp[:,:,1],[R1, B1], [1,2], scatter=False, dx=1.25)
axV.set_xlim([-0.5,4.75])

axV.set_ylabel(r'$R^2$ (Testset)')
axV.text(0.1, 0.95, 'Velocity', transform = axV.transAxes)
axV.text(0.75,0.95, 'Turn', transform = axV.transAxes)
### plotting starts here
gfp = []
for behavior, xoff in zip(['AngleVelocity', 'Eigenworm3'], [2.25, 5.25]):
    scores = []
    for key, marker in zip(['AML18_moving', 'AML175_moving'],['o', "p"]):
        dset = data[key]['analysis']
        keep = []
        for idn in dset.keys():
            results=  dset[idn][flag][behavior]
            try:
                keep.append(np.array([results['scorepredicted'], np.max(results['individualScore'])]))
            except ValueError:
                keep.append(np.array([results['scorepredicted'], 0]))
        # do some plotting
        keep = np.array(keep)
        rnd1 = np.random.rand(len(keep))*0.2
        rnd2 = np.random.rand(len(keep))*0.2
        c = colorCtrl[behavior]
        axV.scatter(xoff+np.zeros(len(keep))+rnd1, keep[:,0], marker = marker,c = c, edgecolor=c, alpha=0.5, s=25)
        axVb.scatter(xoff+np.zeros(len(keep))+rnd1, keep[:,0], marker = marker,c = c, edgecolor=c, alpha=0.5, s=25)

        scores.append(keep)
    gfp.append(np.concatenate(scores, axis=0))
gfp = np.array(gfp)

# now boxplot for the full set
x0 = 2
mkStyledBoxplot(axV, [x0, x0+3],gfp[:,:,0],[N0, N1], ['Velocity', 'Turn'],scatter=False, dx=1.25  )
mkStyledBoxplot(axVb, [x0, x0+3],gfp[:,:,0],[N0, N1], ['Velocity', 'Turn'],scatter=False, dx=1.25  )
axV.set_xlim([-0.5,5.5])
axVb.set_xlim([-0.5,5.5])
axV.axhline(color='k', linestyle=':')
# broken axis stuff
axV.set_ylim([-0.2,axV.get_ylim()[-1]])
axVb.set_ylim([-0.5,-3])
# remove labels and spines
axV.spines['bottom'].set_visible(False)
axV.set_xticks([])
# add fancy linebreaks
d = .015 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=axV.transAxes, color='k', clip_on=False)
axV.plot((-d,d), (-d,+d), **kwargs)
kwargs = dict(transform=axVb.transAxes, color='k', clip_on=False)
axVb.plot((-d,d),(1-d*3,1+d*2), **kwargs)
#
kwargs = dict(transform=axV.transAxes, color='k', clip_on=False)
axV.plot((-d,d), (-d,+d), **kwargs)
kwargs = dict(transform=axVb.transAxes, color='k', clip_on=False)
axVb.plot((-d,d),(1-d*3,1+d*2), **kwargs)

x0= -0.25
axVb.set_xticks([x0, x0+1.75,x0+2.5, x0+3, x0+1.75+3 , x0+5.5])
axVb.set_xticklabels(['G', 'S','Ctrl', 'G', 'S', 'Ctrl'])
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