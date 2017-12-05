# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 2017
Attempt to reproduce Ashley's work and test stability of LASSO versus Clustering+Ridge and elastic net algorithms.
@author: mscholz
"""

import numpy as np
#import fourier_vec as fourier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io
import os
import scipy.linalg as linalg
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
import pyemma.coordinates as coor

from mpl_toolkits.mplot3d.art3d import Line3DCollection

def rankTransform(neuroMap):
    """takes a matrix and transforms values into rank within the colum. ie. neural dynamics: for each neuron
    calculate its rank at the current time."""
    temp = neuroMap.argsort(axis=0)
    rank = temp.argsort(axis=0)
        
    return rank
    
def multicolor(ax,x,y,z,t,c, threedim = True):
    """multicolor plot from francesco."""
    if threedim:
        points = np.array([x,y,z]).transpose().reshape(-1,1,3)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = Line3DCollection(segs, cmap=c)
        lc.set_array(t)
        ax.add_collection3d(lc)
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_ylim(np.min(y),np.max(y))
        ax.set_zlim(np.min(z),np.max(z))
    else:
        points = np.array([x,y]).transpose().reshape(-1,1,2)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = LineCollection(segs, cmap=c)
        lc.set_array(t)
        ax.add_collection(lc)
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_ylim(np.min(y),np.max(y))
        

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')



def qualityCheckBehavior(T, etho, xPos, yPos, vel, pc12, pc3):
    plt.figure()
    ax1 = plt.subplot(321)
    plotEthogram(ax1, T, etho)
    plt.ylabel('Ethogram')
    plt.subplot(322)
    plt.plot(xPos, yPos)
    plt.subplot(323)
    cax1 = plt.imshow(Y, aspect='auto', interpolation='none', extent=[0,T[-1],lenY,0],vmax=1)
    cbar = plt.colorbar(cax1)
    plt.subplot(324)
    plt.plot(vel)
    plt.subplot(325)
    plt.plot(pc12[:,0], pc12[:,1], '-')
    plt.subplot(326)
    plt.plot(pc3)
    plt.tight_layout()

def plotEthogram(ax, T, etho, alpha = 0.5, yVal=1):
    """make a block graph ethogram for elegans behavior"""
    colDict = {-1:'red',0:'k',1:'green',2:'blue'}
    labelDict = {-1:'Reverse',0:'Pause',1:'Forward',2:'Turn'}
    #y1 = np.where(etho==key,1,0)
    for key in colDict.keys():
        plt.fill_between(T, y1=np.ones(len(T))*yVal, where=(etho==key)[:,0], \
        interpolate=False, color=colDict[key], label=labelDict[key], alpha = alpha)
    plt.xlim([min(T), max(T)])
    #plt.ylim([0, 1.1*yVal])
    
    ax.yaxis.set_visible(False)
    plt.legend(ncol=2)

######################################
#
# load and prepare data
#
#####################################
#folder = '../../PanNeuronal/20171017/BrainScanner20171017_184114/' #also good one
folder = '../../PanNeuronal/GoldStandardDatasets/BrainScanner20170610_105634_linkcopy/' #also good one

data = scipy.io.loadmat(folder+'heatData.mat')
print data.keys()

Y = np.array(data['Ratio2'])
# ordering from correlation map
order = np.array(data['cgIdx']).T[0]-1
# crop first 100 seconds due to issues in starting recording
TZero=1200 #in frames
Y = Y[:,TZero:]
# unpack behavior variable
bData = data['behavior'][0][0].T
etho, xPos, yPos, vel, pc12, pc3 = bData
pc3 = pc3[:,0]
etho = etho[TZero:]
xPos = xPos[TZero:]
yPos = yPos[TZero:]
vel = vel[TZero:]
pc12 = pc12[TZero:]
pc3 = pc3[TZero:]

# prep neural data
Y = Y[order]
mask = np.isnan(Y)
Y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Y[~mask])
lenY = len(Y)
T = np.arange(Y.shape[1])/6.
# in behavior there are 6 entries: ethogram, xpos,ypos, v, pc12, pc3
# -1 for reverse, 0 pause, 1 forward, 2 turn
eigenwormplot = 0

Theta = np.unwrap(np.arctan2(pc12[:,1], pc12[:,0]))
velo = savitzky_golay(Theta, window_size=9, order=3, deriv=1, rate=1)
turns = savitzky_golay(pc3, window_size=9, order=3, deriv=1, rate=1)

neuroSmooth = np.array([savitzky_golay(line, window_size=5, order=3, deriv=0, rate=1) for line in Y])

rankM = rankTransform(neuroSmooth)
plt.subplot(211)
cax1 = plt.imshow(rankM, aspect='auto')
plt.colorbar()
plt.subplot(212)
cax1 = plt.imshow(neuroSmooth, aspect='auto', vmax =0.5, vmin=0.1)
plt.colorbar()
plt.show()
neuroSmooth = rankM
######################################
#
# Working with tica
#
#####################################
ticaObj = tica_obj = coor.tica(neuroSmooth.T, lag=2, dim =5, kinetic_map=False)

# plot
#plt.figure(1223)
Y = ticaObj.get_output()[0]+2
#ax1 = plt.subplot2grid((2,2),(0,0))
#plt.plot(T,Y[:,0], label='0')
#plotEthogram(ax1, T, etho, alpha=0.5, yVal = 5)
#ax2 = plt.subplot2grid((2,2),(0,1))
#plt.plot(T,Y[:,1], label='1')
#plotEthogram(ax2, T, etho, alpha=0.5, yVal = 5)
#ax3 = plt.subplot2grid((2,2),(1,0))
#plt.plot(T,Y[:,2], label='2')
#plotEthogram(ax3, T, etho, alpha=0.5, yVal = 5)
#ax4 = plt.subplot2grid((2,2),(1,1))
#plt.plot(T,Y[:,3], label='3')
#plotEthogram(ax4, T, etho, alpha=0.5, yVal = 5)
#plt.ylabel('tica')
#plt.show()
x,y = Y.T[:2]
ax = plt.subplot(111)
multicolor(ax,x,y,None,etho[:,0],cm.viridis, threedim = 0)
#plt.plot( Y[:,0],  Y[:,1], 'ro')
plt.show()
#print len(velo), len(Y[:,1])
#for i in range(2):
#    fig = plt.figure('Correlates{}'.format(i),(3.5,7))
#    plt.subplot(311)
#    plt.scatter(velo, Y[:,i], label='pc1', alpha=0.01)
#    
#    plt.subplot(313)
#    plt.scatter(turns, Y[:,i], label='pc1', alpha=0.01)
#
#
#plt.show()

######################################
#
# neural dimensionality tica
#
#####################################
fig = plt.figure('Manifolds!',(7,7))
ax = fig.gca(projection='3d')
x,y,z = Y.T[:3]
multicolor(ax,x,y,z,etho[:,0],cm.viridis)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
######################################
#
# Working with Eigenworms
#
#####################################

if eigenwormplot:
    fig = plt.figure('Behavior',(7,7))
    alpha = 0.2
    ax1 = plt.subplot(511)
    plotEthogram(ax1, T, etho, alpha, yVal = max(pc12[:,0]))
    plt.plot(T, velo, 'k')
    
    ax2 = plt.subplot(512, sharex=ax1)
    plotEthogram(ax1, T, etho, alpha, yVal = max(pc12[:,0]))
    plt.plot(T, pc12[:,0], 'k')
    plt.ylabel('Eigenworm 1')
    
    ax3 = plt.subplot(513, sharex=ax1)
    plotEthogram(ax1, T, etho, alpha, yVal = max(pc12[:,1]))
    plt.plot(T, pc12[:,1], 'k')
    plt.ylabel('Eigenworm 2')
    
    ax4 = plt.subplot(514, sharex=ax1)
    plotEthogram(ax1, T, etho, alpha, yVal = max(pc3))
    plt.plot(T, pc3, 'k')
    plt.ylabel('Eigenworm 3')
    
    ax5 = plt.subplot(515, sharex=ax1)
    cax1 = plotHeatmap(T, Y)
    plt.xlabel('Time (s)')
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cax1, cax=cbar_ax)
    fig.subplots_adjust(left = 0.1, bottom = 0.05, top =0.95, right=0.8, hspace=0.15)
    plt.show()

######################################
#
# PCA of neural data
#
#####################################
#%%
nComp = 5

#neuroSmooth -=np.mean(neuroSmooth, axis=0)
#neuroSmooth = Y - np.mean(Y, axis=0)
pca = PCA(n_components = nComp)
pcs = pca.fit_transform(neuroSmooth)
print pcs.shape
# order by weight
indices = np.arange(lenY)
indices = np.argsort(pcs[:,0])
pcs = pcs[indices]

fig = plt.figure('PCA',(7,7))
plt.subplot(221)
cax1 = plotHeatmap(T, neuroSmooth[indices])
plt.xlabel('Time (s)')
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(cax1, cax=cbar_ax)
fig.subplots_adjust(left = 0.1, bottom = 0.05, top =0.95, right=0.8, hspace=0.15)

# plot the weights
plt.subplot(222)
rank = np.arange(0, len(pcs))

#weightsum = np.concatenate([np.zeros((lenY,1)),np.cumsum(pcs, axis=1)], axis=1)
#for i in range(4):
#    plt.fill_betweenx(rank, weightsum[:,i], weightsum[:,i+1], step='pre')
plt.fill_betweenx(rank, np.zeros(lenY),pcs[:,0], step='pre')
plt.fill_betweenx(rank, np.zeros(lenY),pcs[:,1], step='pre')


ax3 = plt.subplot(223)
plt.fill_between(np.arange(nComp),pca.explained_variance_ratio_)
ax3.step(np.arange(nComp),np.cumsum(pca.explained_variance_ratio_), where = 'pre')
#ax3.set_ylabel('Explained variance')

ax4 = plt.subplot(224)
for i in range(4):
    ax4.plot(T,0.1*i+pca.components_[i], label=i)
plotEthogram(ax4, T, etho, alpha=0.5, yVal = 0.5)

plt.show()
#%%
#######################################
##
## correlate neural data and behavior
##
######################################
#fig = plt.figure('CorrelatesFwd',(7,7))
#plt.subplot(311)
#plt.scatter(pc12[:,0], pca.components_[0], label='pc1', alpha=0.01)
#plt.subplot(312)
#plt.scatter(pc12[:,0], pca.components_[1], label='pc2', alpha=0.01)
#plt.subplot(313)
#plt.scatter(pc12[:,0], pca.components_[2], label='pc3', alpha=0.01)
#
#fig = plt.figure('Correlatesbwd',(7,7))
#plt.subplot(311)
#plt.scatter(pc12[:,1], pca.components_[0], label='pc1', alpha=0.01)
#plt.subplot(312)
#plt.scatter(pc12[:,1], pca.components_[1], label='pc2', alpha=0.01)
#plt.subplot(313)
#plt.scatter(pc12[:,1], pca.components_[2], label='pc3', alpha=0.01)
#
#
#fig = plt.figure('CorrelatesTurn',(7,7))
#plt.subplot(311)
#plt.scatter(turns, pca.components_[0], label='pc1', alpha=0.01)
#plt.subplot(312)
#plt.scatter(turns, pca.components_[1], label='pc2', alpha=0.01)
#plt.subplot(313)
#plt.scatter(turns, pca.components_[2], label='pc3', alpha=0.01)

#%%
######################################
#
# neural dimensionality
#
#####################################
fig = plt.figure('Manifolds!',(7,7))
ax = fig.gca(projection='3d')
x,y,z = pca.components_[:3]
multicolor(ax,x,y,z,etho[:,0],cm.viridis)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


#%%
######################################
#
# reproject neural dynamics in lower dim space
#
#####################################
#%%
a  = pca.inverse_transform(pcs)
plt.subplot(211)
cax1 = plotHeatmap(T, neuroSmooth[indices])
plt.subplot(212)
cax1 = plotHeatmap(T, a)

plt.show()

######################################
#
# projection on behavioral axes
#
#####################################