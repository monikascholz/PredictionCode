
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
from scipy.optimize import curve_fit
from scipy.special import erf 
# custom pip

#

import makePlots as mp
import dataHandler as dh

from stylesheet import *
# stats
from sklearn.metrics import r2_score



################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML32', 'AML70']:
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

fig = plt.figure('Fig - 4 : Recreating postures', figsize=(9.5, 4.5))
# this gridspec makes one example plot of a heatmap with its PCA
gs1 = gridspec.GridSpec(2, 2, width_ratios = [1,1], height_ratios=[2,1])
gs1.update(left=0.03, right=0.99, wspace=0.35, bottom = 0.1, top=0.99, hspace=0.15)
# second row
gsNL = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs1[1,:], wspace=0.5, hspace=0.5, width_ratios = [1,1,1,1,1,1])

ax3= plt.subplot(gsNL[0,0])
ax1 = plt.subplot(gs1[0,0], aspect='equal')
ax2 = plt.subplot(gs1[0,1], aspect='equal')
# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B']
y0 = 0.99
locations = [(0,y0),  (0.5,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
            horizontalalignment='left',verticalalignment='top',)
            
letters = ['C', 'D', 'E', 'F', 'G', 'H']
y0 = 0.43
x0 = 0.18
locations = [(0,y0),  (x0,y0), (2*x0,y0), (5*x0,y0),(6*x0,y0), (7*x0,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
            horizontalalignment='left',verticalalignment='top',)
# recreate worms
#=============================================================================#
#                           # load eigenworms
#=============================================================================#
eigenwormsFull = dh.loadEigenBasis(filename='utility/Eigenworms.dat', nComp=5, new = True)    
eigenworms = dh.loadEigenBasis(filename='utility/Eigenworms.dat', nComp=3, new = True)    


pc1, pc2, pc3, avTrue, thetaTrue = moving['Behavior']['Eigenworm1'],moving['Behavior']['Eigenworm2'],\
                        moving['Behavior']['Eigenworm3'],  moving['Behavior']['AngleVelocity'],  moving['Behavior']['Theta']
pcs = np.vstack([pc3,pc2, pc1])
# actual centerline
cl= moving['CL']  
#=============================================================================#
# for debugging recreate an existing, approximated shape from 3 eigenworms
#=============================================================================#    
pcsNew, meanAngle, lengths, refPoint = dh.calculateEigenwormsFromCL(cl, eigenworms)
pc3New, pc2New, pc1New = pcsNew
cl2 = dh.calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint)
# transform eigenworms exactly the same way. Otherwise we get some artefacts from nans
r =(pcsNew[2]**2+pcsNew[1]**2)
#r = (pcs[2]**2+pcs[1]**2)
#=============================================================================#
# here we reconstruct from the true angular velocity to check the math. This is smoothed, so we need to compare with this version
#=============================================================================#
xt, yt, zt = dh.recrWorm(avTrue, pc3, thetaTrue, r=r)
pcsR = np.vstack([zt,yt, xt])
clApprox = dh.calculateCLfromEW(pcsR, eigenworms, meanAngle, lengths, refPoint)
#=============================================================================#
# load predicted worm
#=============================================================================#

avP = movingAnalysis['ElasticNet']['AngleVelocity']['output'][:len(pcs[0])]
tP = movingAnalysis['ElasticNet']['Eigenworm3']['output'][:len(pcs[0])]
print 'R2'
print  movingAnalysis['ElasticNet']['AngleVelocity']['score'], movingAnalysis['ElasticNet']['AngleVelocity']['scorepredicted']
print  movingAnalysis['ElasticNet']['Eigenworm3']['score'], movingAnalysis['ElasticNet']['Eigenworm3']['scorepredicted']

################################################
#
#get data
################################################
# weights
flag = 'ElasticNet'
# one example
time = moving['Neurons']['Time']

#avNeurons = moving['Neurons']['ActivityFull'][np.where(np.abs(avWeights))>0]
#tNeurons = moving['Neurons']['ActivityFull'][np.where(np.abs(tWeights))>0]
label = 'AngleVelocity'
splits = movingAnalysis['Training']
train, test = splits[label]['Train'], splits[label]['Test']
t = moving['Neurons']['Time'][test]

#=============================================================================#    
# reconstruct worm from predicted angular velocity.
#=============================================================================#
xP, yP, zP = dh.recrWorm(avP, tP, thetaTrue,r=r)
pcsP = np.vstack([zP,yP, xP])
clPred = dh.calculateCLfromEW(pcsP, eigenworms, meanAngle, lengths, refPoint)
# center around midpoint
originalCMS = np.tile(np.mean(cl2, axis=1)[:,np.newaxis,:], (1,99,1))
clApprox -= originalCMS
cl2 -= originalCMS
clPred -= originalCMS
# non-linearity corrected
#=============================================================================#    
# correct undershoots in turns
#=============================================================================#
indices = np.argsort(tP)
xdata, ydata = tP[indices], pc3[indices]
# bin the data
def nonlinear(x,A):
    return A*np.abs(x*2)

# bin the data
def fitfun(x,A,m,s):
    # taper off towards the ends, otherwise overshoot horribly
    #return -A*(1-np.exp(-(x-m)**2/s**2))*x
    #return A*x
    #return A*np.abs(x)*x
    #return  -A*(1-np.exp(-(x-m)**2/s**2))
    #return A*erf((x-m)/s)#*x
    #
    return (A*(1-np.exp(-(x-m)**2/s**2))+1)*x
#plt.figure()
#plt.scatter(ydata, abs(xdata-ydata),alpha=0.2)    
#plt.show()
p0 = [4,0,3] 

#plt.figure()
#plt.plot(fitfun(np.arange(-10,10,0.1), *p0))
#plt.show()

popt, pcov = curve_fit(fitfun, xdata, ydata, p0)#,bounds=[[-2,-10, -10], [15,10,10]])

#plt.figure()
#plt.plot(fitfun(xdata, *popt))
#plt.show()
print popt
# new tP
tPnew = fitfun(tP, *popt)
xP, yP, zP = dh.recrWorm(avP, tPnew, thetaTrue,r=r)
pcsP = np.vstack([zP,yP, xP])
cl3 = dh.calculateCLfromEW(pcsP, eigenworms, meanAngle, lengths, refPoint)

cl3 -= originalCMS


################################################
#
#first row - posture prediction
#
################################################
# find samples that are starting with a reset
print np.where(test%60==0)[0]
print t[np.where(test%60==0)[0]]
loc1, loc2 = 1090, 610

# plot predicted behaviors and location of postures
ybeh = [10, 20]
#print -time[samplePost[0]]+time[samplePost[-1]]
for behavior, color, cpred, yl, label, align in zip(['AngleVelocity','Eigenworm3' ], \
            [N1, N1], [R1, B1], ybeh, ['Wave speed', 'Turn'], ['center', 'center']):
    beh = moving['Behavior'][behavior][test]
    
    meanb, maxb = np.mean(beh),np.std(beh)
    beh = (beh-meanb)/maxb
    
    behPred = movingAnalysis[flag][behavior]['output'][test]
    behPred = (behPred-meanb)/maxb
    
    ax3.plot(t, beh+yl, color=color)
    ax3.plot(t, behPred+yl, color=cpred)
    #ax1.text(t[test[-1]], yl, \
    #r'$R^2 = {:.2f}$'.format(np.float(movingAnalysis[flag][behavior]['scorepredicted'])), horizontalalignment = 'right')
    ax3.text(t[0]*0.6, yl, label, rotation=90, color=cpred, verticalalignment=align)
ax3.axvspan(time[test[loc1]], time[test[loc1+60]], color=L0, zorder=-10, alpha=0.75)
ax3.axvspan(time[test[loc2]], time[test[loc2+60]], color=L2, zorder=-10, alpha=0.75)

# add scalebar
l =120
y = ybeh[0]*0.7
ax3.plot([t[0], t[0]+l],[y, y], 'k', lw=2)
ax3.text(t[0]+l*0.5,y*0.5, '2 min', horizontalalignment='center')

ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.set_yticks([])
ax3.set_xticks([])

for ax, samplePost, color in zip([ax1, ax2], [test[loc1:loc1+60:6], test[loc2:loc2+60:6]], [L0, L2]):
    #=============================================================================#    
    # calculate mse
    #=============================================================================#
#    plt.figure()
#    mse = np.mean(np.sum((clPred - clApprox)**2, axis=2), axis=1)
#    plt.plot(mse)
#    plt.plot(test, np.abs(beh)*np.max(mse)/10)
#    print np.where(mse<1)
#    plt.show()
    #=============================================================================#    
    # create finite-width worms
    #=============================================================================#
    # offsets to create aspect ratio and spacing in equal proprtion plots
    offsetx = 450
    offsety = 500
    # original worms
    patches = []
    for cindex, cline in enumerate(cl2[samplePost]):
        x,y = cline[:,0], cline[:,1]
        x -=np.mean(x)-offsetx*cindex
        y -=np.mean(y)
        vertices = createWorm(x,y)
        patches.append(mpl.patches.Polygon(vertices, closed=True))
        #ax.plot(x, y)
    p = mpl.collections.PatchCollection(patches, alpha=1, color=color, edgecolor='none')
    ax.add_collection(p)
    
    
    # 3 eigenworm approximate worms
    patches = []
    for cindex, cline in enumerate(clApprox[samplePost]):
        x,y = cline[:,0], cline[:,1]
        x -=np.mean(x)-offsetx*cindex
        y -=np.mean(y)+offsety
        vertices = createWorm(x,y)
        patches.append(mpl.patches.Polygon(vertices, closed=True))
        #ax.plot(x, y)
    p = mpl.collections.PatchCollection(patches, alpha=1, color=color, edgecolor='none')
    ax.add_collection(p)
    
    # predicted worms
    patches = []
    for cindex, cline in enumerate(clPred[samplePost]):
        x,y = cline[:,0], cline[:,1]
        x -=np.mean(x)-offsetx*cindex
        y -=np.mean(y)+2.5*offsety
        vertices = createWorm(x,y)
        patches.append(mpl.patches.Polygon(vertices, closed=True))
        #ax.plot(x, y)
    p = mpl.collections.PatchCollection(patches, alpha=1, color=N0, edgecolor='none')
    ax.add_collection(p)
    
#    # 3 eigenworm approximate worms
#    patches = []
#    for cindex, cline in enumerate(clApprox[samplePost]):
#        x,y = cline[:,0], cline[:,1]
#        x -=np.mean(x)-offsetx*cindex
#        y -=np.mean(y)+offsety
#        vertices = createWorm(x,y)
#        patches.append(mpl.patches.Polygon(vertices, closed=True))
#        #ax.plot(x, y)
#    p = mpl.collections.PatchCollection(patches, alpha=1, color=N1, edgecolor='none')
#    ax.add_collection(p)
    
    # predicted worms - with nonlinearity
    patches = []
    for cindex, cline in enumerate(cl3[samplePost]):
        x,y = cline[:,0], cline[:,1]
        x -=np.mean(x)-offsetx*cindex
        y -=np.mean(y)+3.5*offsety
        
        vertices = createWorm(x,y)
        patches.append(mpl.patches.Polygon(vertices, closed=True))
        #ax.plot(x, y)
    
    p = mpl.collections.PatchCollection(patches, alpha=1, color=N0, edgecolor='none')
    ax.add_collection(p)
    
    # add horizontal line and text
    #ax.axhline(-2*offsetx,color = 'k', linestyle =':')
    ax.text(0.5, 0.45,'predicted', transform = ax.transAxes, horizontalalignment = 'center')
    ax.text(0.5, 1,'real', transform = ax.transAxes, horizontalalignment = 'center')
    # general plot style
    ax.set_xticks(np.arange(0,len(samplePost+1)*offsetx, 2*offsetx))
    ax.set_xticklabels(['{} s'.format(i) for i in range(0, len(samplePost)+1, 2)])
    ax.set_xlim([-300, 4300])
    ax.set_ylim([-3.5*offsety-350, 350])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
# just left plot
ax1.set_yticks(np.linspace(0,-3*offsety, 4))
ax1.set_yticks([0,- offsety, -2.5*offsety, -3.5*offsety])
ax1.set_yticklabels(['original', 'inferred', 'linear', 'non-linear'])
# move to right
moveAxes(ax1, 'right', 0.07)


################################################
#
#second row - non-linearity
#
################################################


ax4 = plt.subplot(gsNL[0,1])

# show non-linearity for one neuron
ax4.scatter(xdata, ydata, color='k', alpha=0.01)
ax4.plot(xdata, tPnew[indices], color=R1, linestyle='--', label='fit')
ax4.set_xlabel('Predicted turn')
ax4.set_ylabel('True turn', labelpad=-10)
moveAxes(ax4, 'right', 0.02)

#ax5 = plt.subplot(gsNL[0,2])
#ax5.plot(np.arange(-10,10,0.1), nonlinear(np.arange(-10,10,0.1),popt[0]), color='k')
#
#ax5.set_ylabel('Nonlinearity')
#ax5.set_xlabel('Turns')
ax6 = plt.subplot(gsNL[0,2])
ax7 = plt.subplot(gsNL[0,3])
ax8 = plt.subplot(gsNL[0,4])
ax9 = plt.subplot(gsNL[0,5])

# compare to linear
ax6.plot(time[test], pc3[test], color=N1, label='true', zorder=-1, linestyle=':')
ax6.plot(time[test], tP[test], color=B1, label='linear')
# compare to nonlinear
ax7.plot(time[test], pc3[test], color=N1, label='true', zorder=-1, linestyle=':')
ax7.plot(time[test], tPnew[test], color=B2, label='non-linear')

r2orig = r2_score(pc3[test], tP[test])
r2nl = r2_score(pc3[test], tPnew[test])
ax6.text(1,0.95,r'$R^2 = {:.2f}$'.format(r2orig), horizontalalignment = 'right', transform = ax6.transAxes)
ax7.text(1,0.95,r'$R^2 = {:.2f}$'.format(r2nl), horizontalalignment = 'right', transform = ax7.transAxes)
#ax6.legend(loc=8, bbox_to_anchor=(-0.1, 0.5))
ax7.spines['left'].set_visible(False)
ax6.spines['left'].set_visible(False)
ax7.set_yticks([])
ax6.set_yticks([])
ax6.plot([time[test][0], time[test][0]], [-15,-5], color=B1, lw=2)
ax6.text(time[test][0]*0.99, -10,'10', color=B1, verticalalignment='center', horizontalalignment='right')
#ax6.set_ylabel('Turn')
ax6.set_xlabel('Time (s)')
ax7.set_xlabel('Time (s)')
# move
moveAxes(ax6, 'right', 0.015)
moveAxes(ax7, 'left', 0.025)
print 'fit', popt[0]

# calculate non-linear fits for many datasets
#=============================================================================#    
# correct undershoots in turns
#=============================================================================#
fitdata = []
r2s = []
for key in ['AML32_moving', 'AML70_chip']:
    dset = data[key]['input']
    res = data[key]['analysis']
    for idn in dset.keys():
        xdata =res[idn]['ElasticNet']['Eigenworm3']['output']
        ydata = dset[idn]['Behavior']['Eigenworm3']
        test = res[idn]['Training']['Eigenworm3']['Test']
        train = res[idn]['Training']['Eigenworm3']['Train']
        indices = np.argsort(xdata[train])
        xdataS= xdata[indices]
        ydataS= ydata[indices]
       
        p0 = [1,1,10] 
#        plt.plot(xdataS, fitfun(xdataS, *popt))
#        plt.scatter(xdataS, ydataS)
#        plt.show()
        popt, pcov = curve_fit(fitfun, xdataS, ydataS, p0)#,  bounds=[[-10,1, -10], [10,4,100]])
        print popt
        fitdata.append(popt[0])
        
#        plt.figure()
#        plt.plot(ydata)
#        plt.plot(xdata)
#        plt.plot(fitfun(ydata, *popt), 'r--')
#        plt.show()
        ax8.plot(np.arange(-10,10, 0.1), fitfun(np.arange(-10,10, 0.1), *popt), color='k', alpha=0.25)
        r2s.append([r2_score( ydata[test],xdata[test]), r2_score(ydata[test],fitfun(xdata[test], *popt))])
print fitdata
#mkStyledBoxplot(ax8, [0,1], np.reshape(fitdata, (1,-1)),[L1], ['A'])
mkStyledBoxplot(ax9, [0,1], np.array(r2s).T,[L1, L3], ['L', 'NL'])
ax9.set_xlim([-0.5,1.5])
ax9.set_ylabel(r'$R^2$')
ax8.set_xlabel('Linear')
ax8.set_ylabel('Non-linear')
plt.show()