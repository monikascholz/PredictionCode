
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
from scipy.optimize import curve_fit
import makePlots as mp
import dataHandler as dh
# deliberate import all!
from stylesheet import *
from scipy import ndimage


def compareReconstructedWorms(cl, eigenworms, avTrue, thetaTrue,pc3, avP, tP, tPnew):
    pcsNew, meanAngle, lengths, refPoint = dh.calculateEigenwormsFromCL(cl, eigenworms)
    pc3New, pc2New, pc1New = pcsNew
    cl2 = dh.calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint)
    cl = cl[:,:-1,:]
    cl = ndimage.gaussian_filter1d(cl, 5, 1)
    # transform eigenworms exactly the same way. Otherwise we get some artefacts from nans
    r =(pcsNew[2]**2+pcsNew[1]**2)
    r = np.repeat(np.median(r), len(r))
    #lengths = 5
    #=============================================================================#
    # here we reconstruct from the true angular velocity to check the math. This is smoothed, so we need to compare with this version
    #=============================================================================#
    xt, yt, zt = dh.recrWorm(avTrue, pc3, thetaTrue, r=r, show=False)
    pcsR = np.vstack([zt,yt, xt])
    clApprox = dh.calculateCLfromEW(pcsR, eigenworms, meanAngle, lengths, refPoint)
    #=============================================================================#    
    # reconstruct worm from predicted angular velocity.
    #=============================================================================#
    xP, yP, zP = dh.recrWorm(avP, tP, thetaTrue,r=r)
    pcsP = np.vstack([zP,yP, xP])
    clPred = dh.calculateCLfromEW(pcsP, eigenworms, meanAngle, lengths, refPoint)
    #=============================================================================#    
    # reconstruct worm from non-linear corrected turns
    #=============================================================================#
    xP, yP, zP = dh.recrWorm(avP, tPnew, thetaTrue,r=r)
    pcsP = np.vstack([zP,yP, xP])
    cl3 = dh.calculateCLfromEW(pcsP, eigenworms, meanAngle, lengths, refPoint)
    #=============================================================================#    
    # reconstruct worm from non-linear corrected turns
    #=============================================================================#
    # center around midpoint
    originalCMS = np.tile(np.mean(cl2, axis=1)[:,np.newaxis,:], (1,99,1))
    clApprox -= originalCMS
    cl -= originalCMS
    cl2 -= originalCMS
    clPred -= originalCMS
    cl3 -= originalCMS
    #=============================================================================#    
    # calculate cms velocity with direction
    #=============================================================================#
    trueV = np.tile(avTrue, (2,1)).T
    trueV[:,0]*=-np.cos(meanAngle)
    trueV[:,1]*=np.sin(meanAngle)
    refPoint = np.cumsum(trueV, axis=0)*25
    #print refPoint.shape
    
    predV = np.tile(avP, (2,1)).T
    predV[:,0]*=-np.cos(meanAngle)
    predV[:,1]*=np.sin(meanAngle)
    refPoint2 = np.cumsum(predV, axis=0)*25

    velocity = np.stack([refPoint, refPoint2+(-250,0)])
#    plt.plot(refPoint[:,0],refPoint[:,1])
#    plt.plot(refPoint2[:,0],refPoint2[:,1])
#    plt.show()
    return cl, cl2,  clPred, cl3, velocity
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
#first row
#
################################################
# select a special dataset - moving AML32
#movingAML32 = 'BrainScanner20170424_105620'#'
movingAML32 = 'BrainScanner20170613_134800'
#movingAML32 = 'BrainScanner20170610_105634'#'


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

fig = plt.figure('Worm crawling', figsize=(8, 4.5), dpi=300)
gs1 = gridspec.GridSpec(1, 2, width_ratios = [1,3])
gs1.update(left=0.35, right=0.99, wspace=0.0, bottom = 0.1, top=0.99, hspace=0.15)
gs1.update(left=0.075, right=0.95, wspace=0.1, bottom = 0.15, top=0.9, hspace=0.15)

flag = 'ElasticNet'
#flag = 'PCAPred'

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
# load predicted worm
#=============================================================================#

avP = movingAnalysis[flag]['AngleVelocity']['output'][:len(pcs[0])]
tP = movingAnalysis[flag]['Eigenworm3']['output'][:len(pcs[0])]
print 'R2'
print  movingAnalysis[flag]['AngleVelocity']['score'], movingAnalysis[flag]['AngleVelocity']['scorepredicted']
print  movingAnalysis[flag]['Eigenworm3']['score'], movingAnalysis[flag]['Eigenworm3']['scorepredicted']
#=============================================================================#
# change reference point to reflect velocity
#=============================================================================# 

#trueV = np.tile(avTrue[np.newaxis], (2,99,1)).T
#predV = np.tile(avP[np.newaxis], (2,99,1)).T
#trueV = np.tile(avTrue, (2,1)).T
#trueV[:,0]*=-np.cos(meanAngle)
#trueV[:,1]*=np.sin(meanAngle)
#refPoint = np.cumsum(trueV, axis=0)*25
##print refPoint.shape
##plt.plot(refPoint[:,0],refPoint[:,1])
##plt.show()
#predV = np.tile(avP, (2,1)).T
#predV[:,0]*=-np.cos(meanAngle)
#predV[:,1]*=np.sin(meanAngle)
#refPoint2 = np.cumsum(predV, axis=0)*25
#velocity = np.stack([refPoint, refPoint2+(500,0)])
#plt.plot(refPoint[:,0],refPoint[:,1])
#plt.plot(refPoint2[:,0],refPoint2[:,1])
#plt.show()
#refPoint = np.zeros(refPoint.shape)
#refPoint2 = np.zeros(refPoint.shape)

tPnew = tP 
cl, cl2, clPred, cl3, velocity = compareReconstructedWorms(cl, eigenworms, avTrue, thetaTrue,pc3, avP, tP, tPnew)


# output of behavior prediction from elastic net
flag = flag
ybeh = [0, -6]
axscheme1 = plt.subplot(gs1[0,0])
axscheme2 = plt.subplot(gs1[0,1],adjustable='box', aspect=0.66)
axscheme2.set_ylim(-800, 500)
axscheme2.set_xlim(-450, 950)
axscheme1.set_title('Sparse linear model', y=1.0)
#axscheme1.set_title('PCA model', y=1.05)


for behavior, color, cpred, yl, label, align in zip(['AngleVelocity','Eigenworm3' ], \
            [N1, N1], [R1, B1], ybeh, ['Velocity', 'Body \n curvature'], ['center', 'center']):
    beh = moving['Behavior'][behavior]
    behPred = movingAnalysis[flag][behavior]['output'][test]
    meanb, maxb = np.nanmean(beh),np.nanstd(beh)
    beh = (beh[test]-meanb)/maxb
    if flag=='PCAPred':
        beh = beh#[test]
    else:
        behPred = (behPred-meanb)/maxb
    
    axscheme1.plot(t, beh+yl, color=color)
    axscheme1.plot(t, behPred+yl, color=cpred)
    axscheme1.text(t[-1], np.max(yl+behPred), \
    r'$R^2 = {:.2f}$'.format(np.float(movingAnalysis[flag][behavior]['scorepredicted'])), horizontalalignment = 'right')


axscheme1.text(t[0]*0.78, 0, 'Velocity', rotation=90, color=R1, verticalalignment='center')
axscheme1.text(t[0]*0.7, -6, 'Body', rotation=90, color=B1, verticalalignment='center')
axscheme1.text(t[0]*0.85, -6, 'curvature', rotation=90, color=B1, verticalalignment='center')
axscheme1.spines['left'].set_visible(False)
axscheme1.set_yticks([])
axscheme1.set_xlabel('Time (s)')
#gs1.tight_layout(fig)

cleanAxes(axscheme2)
axscheme2.grid(color='r', linestyle='-', linewidth=2)
axscheme2.set_facecolor(N2)
axscheme1.set_xlim([t[0], t[-1]])

print velocity.shape
mp.make_animation3(fig, axscheme1,timeActual, axscheme2, cl, clPred+(500,0), frames=test, color = N0, save= 1, fname='crawling.mp4',velocity=velocity)
# animate wor
plt.show()