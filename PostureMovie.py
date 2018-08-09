
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

fig = plt.figure('Worm crawling', figsize=(9.5, 3.5))
gs1 = gridspec.GridSpec(1, 2, width_ratios = [1,4])
gs1.update(left=0.03, right=0.99, wspace=0.15, bottom = 0.1, top=0.99, hspace=0.15)


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
#refPoint = np.zeros(refPoint.shape)
pc3New, pc2New, pc1New = pcsNew
lengths = 5
cl2 = dh.calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint)
#print refPoint.shape
#plt.plot(refPoint[:,0],refPoint[:,1])
#plt.show()

# transform eigenworms exactly the same way. Otherwise we get some artefacts from nans
r =(pcsNew[2]**2+pcsNew[1]**2)
#=============================================================================#
# here we reconstruct from the true angular velocity to check the math. This is smoothed, so we need to compare with this version
#=============================================================================#
xt, yt, zt = dh.recrWorm(avTrue, pc3, thetaTrue, r=r, show=False)
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

p0 = [1,1,3] 

popt, pcov = curve_fit(fitfun, xdata, ydata, p0)#,bounds=[[-2,-10, -10], [15,10,10]])

print popt
# new tP
tPnew = fitfun(tP, *popt)
xP, yP, zP = dh.recrWorm(avP, tPnew, thetaTrue,r=r)
pcsP = np.vstack([zP,yP, xP])
cl3 = dh.calculateCLfromEW(pcsP, eigenworms, meanAngle, lengths, refPoint)
cl3 -= originalCMS



# output of behavior prediction from elastic net
flag = 'ElasticNet'
ybeh = [0, -6]
axscheme1 = plt.subplot(gs1[0,0])
axscheme2 = plt.subplot(gs1[0,1],adjustable='box', aspect=0.66)
axscheme2.set_ylim(-400, 400)
axscheme2.set_xlim(-400, 800)
axscheme1.set_title('Sparse linear model', y=1.05)

for behavior, color, cpred, yl, label, align in zip(['AngleVelocity','Eigenworm3' ], \
            [N1, N1], [R1, B1], ybeh, ['Velocity', 'Turn'], ['center', 'center']):
    beh = moving['Behavior'][behavior]

    meanb, maxb = np.nanmean(beh),np.nanstd(beh)
    beh = (beh[test]-meanb)/maxb
    
    behPred = movingAnalysis[flag][behavior]['output'][test]
    behPred = (behPred-meanb)/maxb
    
    axscheme1.plot(t, beh+yl, color=color)
    axscheme1.plot(t, behPred+yl, color=cpred)
    axscheme1.text(t[-1], np.max(yl+behPred), \
    r'$R^2 = {:.2f}$'.format(np.float(movingAnalysis[flag][behavior]['scorepredicted'])), horizontalalignment = 'right')
    axscheme1.text(t[0]*0.8, yl, label, rotation=90, color=cpred, verticalalignment=align)
axscheme1.spines['left'].set_visible(False)
axscheme1.set_yticks([])
axscheme1.set_xlabel('Time (s)')
gs1.tight_layout(fig)
velocity = np.vstack([avTrue, avP])
trueV = np.tile(avTrue[np.newaxis], (2,99,1)).T
predV = np.tile(avP[np.newaxis], (2,99,1)).T
cleanAxes(axscheme2)
axscheme2.set_facecolor(N2)
axscheme1.set_xlim([t[0], t[-1]])
print clApprox.shape
print trueV.shape
mp.make_animation3(fig, axscheme1,timeActual, axscheme2, clApprox, clPred + (500,0), frames=test, color = N0, save= True, velocity=velocity)
# animate worm
plt.show()