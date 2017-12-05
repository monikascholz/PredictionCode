# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:41:10 2017
load and compare multiple data sets for behavior prediction.
@author: monika scholz
"""
import scipy.io
import os
import numpy as np
import matplotlib.pylab as plt

def loadData(folder):
    """load matlab data."""
    data = scipy.io.loadmat(folder+'heatDataMS.mat')
    # load rotation matrix
    R = np.loadtxt(folder+'../'+'Rotationmatrix.dat')
    # try to use original eigenworms instead of weirdly transformed ones
#    print data.keys()
#    hasPointsTime = data['hasPointsTime']
#    print hasPointsTime
#    # load timing info for highRes
#    timing = scipy.io.loadmat(folder+'hiResData.mat')
#    timing = timing['dataAll'][0][0][3]
#    
#    # load centerline data eigenproj
#    clines = scipy.io.loadmat(folder+'centerline.mat')
#    eigen = clines['eigenProj']
#    pc1, pc2, pc3 = eigen[:3,:]
#    plt.plot(pc1)
#    plt.show(block=True)
#    print len(timing), len(pc1)
    # unpack behavior variables
    etho, xPos, yPos, vel, pc12, pc3 = data['behavior'][0][0].T
    # get from behavior on heatmap data. These are 'corrected' eigenworms
    etho, xPos, yPos, vel, pc12, pc3 = data['behavior'][0][0].T
    # deal with eigenworms
    pc1 = pc12[:,0]
    pc2 = pc12[:,1]
    pc3 = pc3[:,0]
    # Rotate Eigenworms
    pc1,pc2, pc3 = np.array(np.dot(R, np.vstack([pc1, pc2, pc3])))
    #mask nans in eigenworms by interpolation
    mask1 = np.isnan(pc1)
    mask2 = np.isnan(pc2)
    mask3 = np.isnan(pc3)
    if np.any(mask1):
        pc1[mask1] = np.interp(np.flatnonzero(mask1), np.flatnonzero(~mask1), pc1[~mask1])
    if np.any(mask2):
        pc2[mask2] = np.interp(np.flatnonzero(mask2), np.flatnonzero(~mask2), pc2[~mask2])
    if np.any(mask3):
        pc3[mask3] = np.interp(np.flatnonzero(mask3), np.flatnonzero(~mask3), pc2[~mask3])
    theta = np.unwrap(np.arctan2(pc2, pc1))
    velo = savitzky_golay(theta, window_size=17, order=5, deriv=1, rate=1)
    pc3 = savitzky_golay(pc3, window_size=9, order=5)
#    if flag['heatmap']=='ratio':
#        Y = np.array(data['Ratio2'])
#    if flag['heatmap']=='raw':
#        Y = np.array(data['G2'])/np.array(data['R2'])
#    else:# by default load phtocorrected but not otherwise corrected data
    #Y= np.array(data['gPhotoCorr'])/np.array(data['rPhotoCorr'])
    Y = np.array(data['Ratio2'])
    # ordering from correlation map/hie
    order = np.array(data['cgIdx']).T[0]-1
    # unpack neuron position (only one frame, randomly chosen)
    try:
        neuroPos = data['XYZcoord'][order].T
    except KeyError:
        neuroPos = []
        print 'No neuron positions:', folder
    # prep neural data by masking nans
    # store relevant indices
    nonNan = np.arange(Y.shape[1])#np.where(np.all(np.isfinite(Y),axis=0))[0]
    Y = Y[order]
    mask = np.isnan(Y)
    Y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Y[~mask])
    Y = rankTransform(Y)
    # smooth with small window size
    #Y = np.array([savitzky_golay(line, window_size=15, order=3) for line in Y])
    
   
    # create a time axis in seconds
    T = np.arange(Y.shape[1])/6.
    # create a dictionary structure of these data
    dataDict = {}
    dataDict['Behavior'] = {}
    tmpData = [vel[:,0], pc1, pc2, pc3, velo, etho]
    for kindex, key in enumerate(['CMSVelocity', 'Eigenworm1', 'Eigenworm2', 'Eigenworm3', 'AngleVelocity', 'Ethogram']):
        dataDict['Behavior'][key] = tmpData[kindex][nonNan]
    dataDict['Neurons'] = {}
    dataDict['Neurons']['Time'] = T[nonNan]
    dataDict['Neurons']['Activity'] = Y[:,nonNan]
    dataDict['Neurons']['rankActivity'] = rankTransform(Y)[:,nonNan]
    dataDict['Neurons']['Positions'] = neuroPos
    return dataDict
    
    
def loadMultipleDatasets(dataLog, pathTemplate):
    """load matlab files containing brainscanner data. 
    string dataLog: file containing Brainscanner names with timestamps e.g. BrainScanner20160413_133747.
    path pathtemplate: relative or absoluet location of the dataset with a formatter replacing the folder name. e.g.
                        GoldStandardDatasets/{}_linkcopy

    return: dict of dictionaries with neuron and behavior data
    """
    datasets={}
    for lindex, line in enumerate(np.loadtxt(dataLog, dtype=str, ndmin = 2)):
        folder = pathTemplate.format(line[0])
        datasets[line[0]] = loadData(folder)
    return datasets

def loadNeuronPositions(filename):
    x = scipy.io.loadmat(filename)['x']
    y = scipy.io.loadmat(filename)['y']
    z = scipy.io.loadmat(filename)['z']
    neuronID = scipy.io.loadmat(filename)['ID']
    # remove non-head neurons
    indices = np.where((y<-2.3)&(x<0.1))
    return np.stack((neuronID[indices],x[indices],y[indices],z[indices]))

def rankTransform(neuroMap):
    """takes a matrix and transforms values into rank within the colum. ie. neural dynamics: for each neuron
    calculate its rank at the current time."""
    temp = neuroMap.argsort(axis=0)
    rank = temp.argsort(axis=0)
    return rank


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