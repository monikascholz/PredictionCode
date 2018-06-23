
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
for typ in ['AML32']:
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
gs1 = gridspec.GridSpec(2, 4, width_ratios = [1,1,1,1])
gs1.update(left=0.01, right=0.98, wspace=0.45, bottom = 0.1, top=0.97, hspace=0.5)

# add a,b,c letters, 9 pt final size = 18pt in this case
#letters = ['A', 'B', 'C']
#y0 = 0.99
#locations = [(0,y0),  (0.47,y0), (0.76,y0)]
#for letter, loc in zip(letters, locations):
#    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
#            horizontalalignment='left',verticalalignment='top',)
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


ax1= plt.subplot(gs1[0,0])
ax2 = plt.subplot(gs1[0,1:], aspect='equal')
ax3= plt.subplot(gs1[1,0], sharex=ax1)
ax4 = plt.subplot(gs1[1,1:], aspect='equal')
#ax5= plt.subplot(gs1[2,0], sharex=ax1)
#ax6 = plt.subplot(gs1[2,1:], aspect='equal')

################################################
#
#first row - posture prediction
#
################################################
# find samples that are starting with a reset
print np.where(test%60==0)[0]
print t[np.where(test%60==0)[0]]
loc1, loc2 = 1090, 610

for ax1, ax2, samplePost in zip([ax1, ax3], [ax2, ax4], [test[loc1:loc1+60:6], test[loc2:loc2+60:6]]):
    ybeh = [10, 15]
    print -time[samplePost[0]]+time[samplePost[-1]]
    for behavior, color, cpred, yl, label in zip(['AngleVelocity','Eigenworm3' ], \
                [N1, N1], [R1, B1], ybeh, ['Wave speed', 'Turn']):
        beh = moving['Behavior'][behavior][test]
        
        meanb, maxb = np.mean(beh),np.std(beh)
        beh = (beh-meanb)/maxb
        
        behPred = movingAnalysis[flag][behavior]['output'][test]
        behPred = (behPred-meanb)/maxb
        
        ax1.plot(t, beh+yl, color=color)
        ax1.plot(t, behPred+yl, color=cpred)
        #ax1.text(t[test[-1]], yl, \
        #r'$R^2 = {:.2f}$'.format(np.float(movingAnalysis[flag][behavior]['scorepredicted'])), horizontalalignment = 'right')
        ax1.text(t[-1]*1.1, yl, label, rotation=90, color=cpred, verticalalignment='center')
    ax1.axvspan(time[samplePost[0]], time[samplePost[-1]], color=N0, zorder=-10, alpha=0.75)
    
    # add scalebar
    l =120
    y = ybeh[0]*0.9
    ax1.plot([t[0], t[0]+l],[y, y], 'k', lw=2)
    ax1.text(t[0]+l*0.5,y*0.8, '2 min', horizontalalignment='center')

    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_yticks([])
    ax1.set_xticks([])

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
    
    offsetx = 450
    # original worms
    patches = []
    for cindex, cline in enumerate(cl2[samplePost]):
        x,y = cline[:,0], cline[:,1]
        x -=np.mean(x)-offsetx*cindex
        y -=np.mean(y)
        vertices = createWorm(x,y)
        patches.append(mpl.patches.Polygon(vertices, closed=True))
        #ax2.plot(x, y)
    p = mpl.collections.PatchCollection(patches, alpha=1, color=N2, edgecolor='none')
    ax2.add_collection(p)
    
    offsety = 500
    # 3 eigenworm approximate worms
    patches = []
    for cindex, cline in enumerate(clApprox[samplePost]):
        x,y = cline[:,0], cline[:,1]
        x -=np.mean(x)-offsetx*cindex
        y -=np.mean(y)+offsety
        vertices = createWorm(x,y)
        patches.append(mpl.patches.Polygon(vertices, closed=True))
        #ax2.plot(x, y)
    p = mpl.collections.PatchCollection(patches, alpha=1, color=N1, edgecolor='none')
    ax2.add_collection(p)
    
    # predicted worms
    patches = []
    for cindex, cline in enumerate(clPred[samplePost]):
        x,y = cline[:,0], cline[:,1]
        x -=np.mean(x)-offsetx*cindex
        y -=np.mean(y)+2*offsety
        vertices = createWorm(x,y)
        patches.append(mpl.patches.Polygon(vertices, closed=True))
        #ax2.plot(x, y)
    p = mpl.collections.PatchCollection(patches, alpha=1, color=N0, edgecolor='none')
    ax2.add_collection(p)
    
    
    # non-linearity corrected
    #=============================================================================#    
    # correct undershoots in turns
    #=============================================================================#
    indices = np.argsort(tP)
    xdata, ydata = tP[indices], pc3[indices]
    # bin the data
    def fitfun(x,A,m,s):
        return A*np.abs(x*2)*x
        #return -A*erf((np.abs(x)-m)/s)*x
    p0 = [1,0,0.1] 
    popt, pcov = curve_fit(fitfun, xdata, ydata, p0)
    # new tP
    tPnew = fitfun(tP, *popt)
    xP, yP, zP = dh.recrWorm(avP, tPnew, thetaTrue,r=r)
    pcsP = np.vstack([zP,yP, xP])
    cl3 = dh.calculateCLfromEW(pcsP, eigenworms, meanAngle, lengths, refPoint)
    
    cl3 -= originalCMS
    # 3 eigenworm approximate worms
    patches = []
    for cindex, cline in enumerate(clApprox[samplePost]):
        x,y = cline[:,0], cline[:,1]
        x -=np.mean(x)-offsetx*cindex
        y -=np.mean(y)+offsety
        vertices = createWorm(x,y)
        patches.append(mpl.patches.Polygon(vertices, closed=True))
        #ax2.plot(x, y)
    p = mpl.collections.PatchCollection(patches, alpha=1, color=N1, edgecolor='none')
    ax2.add_collection(p)
    
    # predicted worms
    patches = []
    for cindex, cline in enumerate(cl3[samplePost]):
        x,y = cline[:,0], cline[:,1]
        x -=np.mean(x)-offsetx*cindex
        y -=np.mean(y)+3*offsety
        vertices = createWorm(x,y)
        patches.append(mpl.patches.Polygon(vertices, closed=True))
        #ax2.plot(x, y)
    p = mpl.collections.PatchCollection(patches, alpha=1, color=N0, edgecolor='none')
    ax2.add_collection(p)
    ax2.set_xticks(np.linspace(0,len(samplePost+1)*offsetx, len(samplePost)+1 ))
    ax2.set_xticklabels(['{} s'.format(i) for i in range(len(samplePost)+1)])
    ax2.set_xlim([-300, 4300])
    ax2.set_ylim([350, -1700])
    ax2.set_yticks(np.linspace(0,-3*offsety, 4))
    ax2.set_yticklabels(['original', 'approximated', 'predicted', 'non-linearity'])
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)


    
plt.show()