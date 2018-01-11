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
import scipy.interpolate
from scipy.signal import medfilt
from sklearn import preprocessing
import makePlots as mp
from scipy.ndimage.filters import gaussian_filter1d
import h5py

def calcFFT(data, time_step=1/6.):
    """plot frequency of data"""
    fft = []
    for line in data:
        #line -= np.mean(line)
        ps = np.abs(np.fft.fft(line))**2
        
        
        freqs = np.fft.fftfreq(line.size, time_step)
        idx = np.argsort(freqs)
    
        
        fft.append(ps[idx])
    return freqs[idx], fft

def makeEthogram(anglevelocity, pc3):
    """use rotated Eigenworms to create a new Ethogram."""
    etho = np.zeros((len(anglevelocity),1))
    # set forward and backward
    etho[np.where(anglevelocity>0)] = 1
    etho[np.where(anglevelocity<=0)] = -1
    # overwrite this in case of turns
    etho[np.abs(pc3)>8] = 2
    return etho

def loadPoints(folder, straight = True):
    """get tracked points from Pointfile."""
    points = np.squeeze(scipy.io.loadmat(folder+'pointStatsNew.mat')['pointStatsNew'])
    print points.shape
    print len(points[10]), [len(p[0]) for p in points[10]]
    if straight:
        return [p[0] for p in points]
    else:
        return [p[1] for p in points]

def loadCenterlines(folder):
    """get centerlines from centerline.mat file"""
    tmp = scipy.io.loadmat(folder+'heatDataMS.mat')
    clTime = np.squeeze(tmp['clTime']) # 50Hz centerline times
    volTime =  np.squeeze(tmp['hasPointsTime'])# 6 vol/sec neuron times
    #print volTime.shape
    clIndices = np.rint(np.interp(volTime, clTime, np.arange(len(clTime))))
    
    #cl = scipy.io.loadmat(folder+'centerline.mat')['centerline']
    
    cl = np.rollaxis(scipy.io.loadmat(folder+'centerline.mat')['centerline'], 2,0)
    #
    #if wormcentered:
    wc = np.rollaxis(scipy.io.loadmat(folder+'centerline.mat')['wormcentered'], 1,0)
    # reduce to volume time
    clNew = cl[clIndices.astype(int)]
#    for cl in clNew[::10]:
#        plt.plot(cl[:,0], cl[:,1])
#    plt.show()
    print 'Done loading centerlines'
    return clNew, wc
    
def transformEigenworms(pc1, pc2, pc3, dataPars):
    """smooth Eigenworms and calculate associated metrics like velocity."""
    theta = np.unwrap(np.arctan2(pc2, pc1))
    #velo = savitzky_golay(theta, window_size=dataPars['savGolayWindow'], order=3, deriv=1, rate=1)
    velo = gaussian_filter1d(theta, dataPars['savGolayWindow'], order=1)
    
    # median filter the velocity and pca components 
    if dataPars['medianWindow'] < 3:
        return pc1, pc2, pc3, velo, theta
    #velo = scipy.signal.medfilt(velo, dataPars['medianWindow'])
    
#    pc1 = scipy.signal.medfilt(pc1, dataPars['medianWindow'])
#    pc2 = scipy.signal.medfilt(pc2, dataPars['medianWindow'])
#    pc3 = scipy.signal.medfilt(pc3, dataPars['medianWindow'])
    pc1 = gaussian_filter1d(pc1, dataPars['medianWindow'])
    pc2 = gaussian_filter1d(pc2, dataPars['medianWindow'])
    pc3 = gaussian_filter1d(pc3, dataPars['medianWindow'])
    
    return pc1, pc2, pc3, velo, theta
    

def loadData(folder, dataPars):
    """load matlab data."""
    data = scipy.io.loadmat(folder+'heatDataMS.mat')
    
    # unpack behavior variables
    ethoOrig, xPos, yPos, vel, pc12, pc3 = data['behavior'][0][0].T
    
    # deal with eigenworms
    pc1 = pc12[:,0]
    pc2 = pc12[:,1]
    pc3 = pc3[:,0]
    # Rotate Eigenworms
    if dataPars['rotate']:
        # load rotation matrix
        # set rotate to False when doing rotation matrix calculation
        R = np.loadtxt(folder+'../'+'Rotationmatrix.dat')
        pc1,pc2, pc3 = np.array(np.dot(R, np.vstack([pc1, pc2, pc3])))
    #mask nans in eigenworms by linear interpolation
    mask1 = np.isnan(pc1)
    mask2 = np.isnan(pc2)
    mask3 = np.isnan(pc3)
    if np.any(mask1):
        pc1[mask1] = np.interp(np.flatnonzero(mask1), np.flatnonzero(~mask1), pc1[~mask1])
    if np.any(mask2):
        pc2[mask2] = np.interp(np.flatnonzero(mask2), np.flatnonzero(~mask2), pc2[~mask2])
    if np.any(mask3):
        pc3[mask3] = np.interp(np.flatnonzero(mask3), np.flatnonzero(~mask3), pc3[~mask3])
    
    # do Eigenworm transformations and calculate velocity etc.
#    # median filter the Eigenworms
    pc1, pc2, pc3, velo, theta = transformEigenworms(pc1, pc2, pc3, dataPars)
    ## recalculate velocity from position
    #vel = np.squeeze(np.sqrt((np.diff(xPos, axis=0)**2 + np.diff(yPos, axis=0)**2))/6.)
    #vel = np.pad(vel, (1,0), 'constant')
    # ethogram redone
    etho = makeEthogram(velo, pc3)
    #etho = ethoOrig
    T = np.arange(pc1.shape[0])/6.
#    print T, etho.shape, ethoOrig.shape
#    mp.plotEthogram(ax, T, ethoOrig, alpha = 0.5, yValMax=1, yValMin=0, legend=0)
#    ax = plt.subplot(212)
#    mp.plotEthogram(ax, T, etho, alpha = 0.5, yValMax=1, yValMin=0, legend=0)
#    plt.show()
    #print vel.shape, xPos.shape
#    else:# by default load phtocorrected but not otherwise corrected data
    R = np.array(data['rPhotoCorr'])
    G = np.array(data['gPhotoCorr'])
    mask = np.isnan(R)
    R[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), R[~mask])
    mask = np.isnan(G)
    G[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), G[~mask])
    
    
    
    # smooth with GCamp6 halftime = 1s
    #GS = np.array([savitzky_golay(line, window_size=dataPars['savGolayWindowGCamp'], order=11) for line in G])
    #RS = np.array([savitzky_golay(line, window_size=dataPars['savGolayWindowGCamp'], order=11) for line in R])
    RS =np.array([gaussian_filter1d(line,dataPars['windowGCamp']) for line in R])       
    GS =np.array([gaussian_filter1d(line,dataPars['windowGCamp']) for line in G])       
    YR = GS/RS
    
    #Y =  np.array([medfilt(line, 21) for line in Y])
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
    nonNan = np.arange(0, YR.shape[1])
    
    #nonNan  = np.where(np.any(np.isfinite(data['rPhotoCorr']),axis=0))[0]
    #print nonNan
    YR = YR[order]
    
#    # make values nicer
    #Y -= np.nanmin(Y, axis=0)
    #Y = (Y-np.mean(Y, axis=0))/np.nanmax(Y, axis=0)
    # smooth with small window size
    #Y = np.array([savitzky_golay(line, window_size=17, order=3) for line in Y])
    #Y =  preprocessing.scale(Y.T).T
    #quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
    #Y = quantile_transformer.fit_transform(Y.T).T
#    plt.imshow(Y, aspect='auto')
#    plt.colorbar()
#    plt.show()
    # long-window size smoothing filter to subtract overall fluctuation in SNR
    wind = 90
    mean = np.mean(rolling_window(np.mean(YR,axis=0), window=2*wind), axis=1)
    #pad with normal mean in front to correctly center the mean values
    mean = np.pad(mean, (wind,0), mode='constant', constant_values=(np.mean(np.mean(YR,axis=0)[:wind])))[:-wind]
    # do the same in the end
    mean[-wind:] = np.repeat(np.mean(np.mean(YR,axis=0)[:-wind]), wind)
   
    YN = YR-mean
#    m, s = np.mean(Y, axis=0), np.std(Y, axis=0)
#    plt.subplot(211)
#    plt.plot(m, 'r', label='mean')
#    plt.plot(mean, label='rolling mean')
#    plt.fill_between(range(len(m)), m-s, m+s, alpha=0.5, label='stdev')
#    plt.ylabel('R/R0')
#    plt.legend()
#    Y -= mean
#
#    plt.subplot(212)
#    
#    plt.plot(np.mean(Y, axis=0)/s, label='rolling Z-score')
#    plt.plot(np.mean(Y, axis=0), label ='mean subtracted')
#    plt.ylabel('<Signal>/Stdev(Signal)')
#    plt.xlabel('frames')
#    plt.tight_layout()
#    plt.legend()
#    plt.show()
    # zscore values 
    Y =  preprocessing.scale(YN.T).T
    # create a time axis in seconds
    T = np.arange(Y.shape[1])/6.
    # redo time axis in seconds for nan issues
    T = np.arange(Y[:,nonNan].shape[1])/6.
    
    if 0:
        #### show what pipeline does
        titles= ['Bleaching corrected', 'Gaussian filter $\sigma=5$', 'Rolling mean (30 s) ', 'Z score']
        for i, hm in enumerate([G[order]/R[order],YR, YN, Y]):
            ax=plt.subplot(2,2,i+1)
            low, high = np.percentile(hm, [2.28, 97.72])#[ 15.87, 84.13])
            ax.set_title(titles[i])
            cax1 = ax.imshow( hm, aspect='auto', interpolation='none', origin='lower',extent=[0,T[-1],len(Y),0],vmax=high, vmin=low)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel("Neuron")
        plt.tight_layout()
        plt.show()
        
        for i, hm in enumerate([G[order]/R[order],YR, YN, Y]):
            ax=plt.subplot(2,2,i+1)
            f, ps = calcFFT(hm, time_step=1/6.)
            ax.set_title(titles[i])
            m, s = np.mean(ps, axis=0), np.std(ps, axis=0)
            ax.plot(f, m, 'r')
            #ax.fill_between(f, m-s, m+s, alpha=0.2, color='r')
            ax.set_yscale('log',nonposy='clip')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel("Power spectrum")
        plt.tight_layout()
        plt.show()
    # create a dictionary structure of these data
    dataDict = {}
    dataDict['Behavior'] = {}
    tmpData = [vel, pc1, pc2, pc3, velo, theta, etho, xPos, yPos]
    for kindex, key in enumerate(['CMSVelocity', 'Eigenworm1', 'Eigenworm2', 'Eigenworm3',\
                'AngleVelocity','Theta', 'Ethogram', 'X', 'Y']):
        dataDict['Behavior'][key] = tmpData[kindex][nonNan]
    dataDict['Neurons'] = {}
    dataDict['Neurons']['Time'] =  np.arange(Y[:,nonNan].shape[1])/6.#T[nonNan]
    dataDict['Neurons']['Activity'] = Y[:,nonNan]
    dataDict['Neurons']['rankActivity'] = rankTransform(Y)[:,nonNan]
    dataDict['Neurons']['Positions'] = neuroPos
    return dataDict
    
    
def loadMultipleDatasets(dataLog, pathTemplate, dataPars):
    """load matlab files containing brainscanner data. 
    string dataLog: file containing Brainscanner names with timestamps e.g. BrainScanner20160413_133747.
    path pathtemplate: relative or absoluet location of the dataset with a formatter replacing the folder name. e.g.
                        GoldStandardDatasets/{}_linkcopy

    return: dict of dictionaries with neuron and behavior data
    """
    datasets={}
    for lindex, line in enumerate(np.loadtxt(dataLog, dtype=str, ndmin = 2)):
        folder = pathTemplate.format(line[0])
        datasets[line[0]] = loadData(folder, dataPars)
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

def rolling_window(a, window):
    a = np.pad(a, (0,window), mode="constant", constant_values=(np.nan,))
    shape = a.shape[:-1] + (a.shape[-1] - window, window)
    strides = a.strides + (a.strides[-1],)
    
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    

def saveDictToHDF(filePath, d):
    f = h5py.File(filePath,'w')
    for fnKey in d.keys():
        for amKey in d[fnKey].keys():
            for attKey in d[fnKey][amKey].keys():
                if type(d[fnKey][amKey][attKey]) is not dict:
                    dataPath = '/%s/%s/%s'%(fnKey,amKey,attKey)
                    f.create_dataset(dataPath,data=d[fnKey][amKey][attKey])
                else:
                    for bKey in d[fnKey][amKey][attKey].keys():
                        
                        dataPath = '/%s/%s/%s/%s'%(fnKey,amKey,attKey,bKey)
                        f.create_dataset(dataPath,data=d[fnKey][amKey][attKey][bKey])
    f.close()
    return

def loadDictFromHDF(filePath):
    f = h5py.File(filePath,'r')
    d = {}
    for fnKey in f.keys():
        d[fnKey] = {}
        for amKey in f[fnKey].keys():
            d[fnKey][amKey] = {}
            for attKey in f[fnKey][amKey].keys():
                if isinstance(f[fnKey][amKey][attKey], h5py.Dataset):
                    d[fnKey][amKey][attKey] = f[fnKey][amKey][attKey][...]
                else:
                    d[fnKey][amKey][attKey] = {}
                    for bKey in f[fnKey][amKey][attKey].keys():
                        d[fnKey][amKey][attKey][bKey] = f[fnKey][amKey][attKey][bKey][...]
                        
    f.close()
    return d

