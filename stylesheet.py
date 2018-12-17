# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 09:35:52 2018
overall style sheet for prediction figures
@author: monika
"""
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

from scipy.ndimage.filters import gaussian_filter1d
################################################
#
# define colors
#
################################################
mpl.rc('font', **{'sans-serif' : 'FiraSans','family' : 'sans-serif'})
mpl.rc('text.latex', preamble='\usepackage{sfmath}')
mpl.rcParams['image.cmap'] = 'viridis'

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
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.labelsize"]=  18
mpl.rcParams["xtick.labelsize"]=  18
mpl.rcParams["ytick.labelsize"]=  18
mpl.rcParams["axes.labelpad"] = 0


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
# ethocmap
ethocmap = mpl.colors.ListedColormap([mpl.colors.to_rgb(R1), mpl.colors.to_rgb(N1), mpl.colors.to_rgb('#f0a202ff'), mpl.colors.to_rgb(B1)], name='etho', N=None)
ethobounds=[-1,0,1,2, 3]
ethonorm = mpl.colors.BoundaryNorm(ethobounds, ethocmap.N)
colDict = {-1:R1, 0: N1, 1:L3, 2:B1}
labelDict = {-1:'Rev',0:'Pause',1:'Fwd',2:'Turn'}
#=============================================================================#
#                           moving axes
#=============================================================================#

def moveAxes(ax, action, step ):
    if action =='left':
        pos = ax.get_position().get_points()
        pos[:,0] -=step
        
    if action =='right':
        pos = ax.get_position().get_points()
        pos[:,0] +=step
        
    if action =='down':
        pos = ax.get_position().get_points()
        pos[:,1] -=step
    if action =='up':
        pos = ax.get_position().get_points()
        pos[:,1] +=step
    if action =='scale':
        pos = ax.get_position().get_points()
        pos[1,:] +=step/2.
        pos[0,:] -=step/2.
    if action =='scaley':
        pos = ax.get_position().get_points()
        pos[1,1] +=step/2.
        pos[0,1] -=step/2.
    if action =='scalex':
        pos = ax.get_position().get_points()
        pos[1,0] +=step/2.
        pos[0,0] -=step/2.
        
    posNew = mpl.transforms.Bbox(pos)
    ax.set_position(posNew)

#=============================================================================#
#                           clean away spines
#=============================================================================#
def cleanAxes(ax, where='all'):
    '''remove plot spines, ticks, and labels. Either removes both, left or bottom axes.'''
    if where=='all':
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
    if where=='x':
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
    if where=='y':
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
    else:
        print 'Command not found. Use "x" or "y" or "all"'
    
    
def alignAxes(ax1, ax2, where='x'):
    """move axes such that the x or y corners align. Reference is ax1, ax2 gets moved."""
    if where =='xspan':
        x0 = ax1.get_position().get_points()[0][0]
        x1 = ax1.get_position().get_points()[1][0]
        pos = ax2.get_position().get_points()
        pos[0][0] = x0
        pos[1][0] = x1
        ax2.set_position(mpl.transforms.Bbox(pos))  
    if where =='yspan':
        y0 = ax1.get_position().get_points()[0][1]
        y1 = ax1.get_position().get_points()[1][1]
        pos = ax2.get_position().get_points()
        pos[0][1] = y0
        pos[1][1] = y1
        ax2.set_position(mpl.transforms.Bbox(pos))  
    if where =='x':
        x0 = ax1.get_position().get_points()[0][0]
        pos = ax2.get_position().get_points()
        diffx = pos[0][0]-x0
        pos[0][0] = x0
        pos[1][0] -= diffx
        ax2.set_position(mpl.transforms.Bbox(pos))  
    if where =='y':
        y0 = ax1.get_position().get_points()[0][1]
        y1 = ax1.get_position().get_points()[1][1]
        pos = ax2.get_position().get_points()
        diffy = pos[0][1]-y0
        pos[0][1] = y0
        pos[1][1] -= diffy
        ax2.set_position(mpl.transforms.Bbox(pos))
        
    else:
        print 'specify alignment, either enter "x" or "y"'
    
#=============================================================================#
#                           plot normal plots
#=============================================================================#
def plotEthogram(ax, T, etho, alpha = 0.5, yValMax=1, yValMin=0, legend=0):
    """make a block graph ethogram for elegans behavior"""
    #colDict = {-1:'red',0:'k',1:'green',2:'blue'}
    #labelDict = {-1:'Reverse',0:'Pause',1:'Forward',2:'Turn'}
    #y1 = np.where(etho==key,1,0)
    
    for key in colDict.keys():
        where = np.squeeze((etho==key))
#        if len((etho==key))==0:
#            
#            continue
        ax.fill_between(T, y1=np.ones(len(T))*yValMin, y2=np.ones(len(T))*yValMax, where=where, \
        interpolate=False, color=colDict[key], label=labelDict[key], alpha = alpha)
    ax.set_xlim([min(T), max(T)])
    ax.set_ylim([yValMin, yValMax])
    ax.set_xlabel('Time (s)')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if legend:
        ax.legend(ncol=2)


def plotHeatmap(T, Y,  ax, vmin=-2, vmax=2):
    """nice looking heatmap for neural dynamics."""
    cax1 = ax.imshow(Y, aspect='auto', interpolation='none', origin='lower',extent=[T[0],T[-1],len(Y),0],vmax=vmax, vmin=vmin)
    ax.set_xlabel('Time (s)')
    ax.set_yticks(np.arange(0, len(Y),25))
    ax.set_ylabel("Neuron")
    return cax1

def multicolor(ax,x,y,z,t,c, threedim = True, etho = False, cg = 1):
    """multicolor plot modified from francesco."""
    lw = 1
    x = x[::cg]
    y = y[::cg]
    if threedim:
        z = z[::cg]
    t = t[::cg]
    if threedim:
        points = np.array([x,y,z]).transpose().reshape(-1,1,3)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = Line3DCollection(segs, cmap=c, lw=lw)
        if etho:
            lc = Line3DCollection(segs, cmap=c, lw=lw, norm=ethonorm)
        lc.set_array(t)
        ax.add_collection3d(lc)
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_ylim(np.min(y),np.max(y))
        ax.set_zlim(np.min(z),np.max(z))
    else:
        points = np.array([x,y]).transpose().reshape(-1,1,2)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = LineCollection(segs, cmap=c, lw=lw)
        if etho:
            lc = LineCollection(segs, cmap=c, lw=lw, norm=ethonorm)
        lc.set_array(t)
        ax.add_collection(lc)
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_ylim(np.min(y),np.max(y))
    return lc


def mkStyledBoxplot(ax, x_data, y_data, clrs, lbls, scatter = True, rotate=True, dx=None) : 
    """nice boxplots with scatter"""
    if dx==None:
        dx = np.min(np.diff(x_data))
    lw = 1.5
    for xd, yd, cl in zip(x_data, y_data, clrs) :
       
        bp = ax.boxplot(yd, positions=[xd], widths = 0.2*dx, \
                        notch=False, patch_artist=True)
        plt.setp(bp['boxes'], edgecolor=cl, facecolor=cl, \
             linewidth=1, alpha=0.4)
        plt.setp(bp['whiskers'], color=cl, linestyle='-', linewidth=lw, alpha=1.0)    
        for cap in bp['caps']:
            cap.set(color=cl, linewidth=lw)       
        for flier in bp['fliers']:
            flier.set(marker='+', color=cl, alpha=1.0)            
        for median in bp['medians']:
            median.set(color=cl, linewidth=lw) 
        jitter = (np.random.random(len(yd)) - 0.5)*dx / 20 
        dotxd = [xd - 0.25*dx]*len(yd) + jitter
        if scatter:
            # make alpha stronger
            ax.plot(dotxd, yd, linestyle='None', marker='o', color=cl, \
                    markersize=3, alpha=0.5)  

    ax.set_xticks(x_data)
    ax.yaxis.set_ticks_position('left') # turn off right ticks
    ax.xaxis.set_ticks_position('bottom') # turn off top ticks
    ax.get_xaxis().set_tick_params(direction='out')
    ax.patch.set_facecolor('white') # ('none')
    if rotate:
        ax.set_xticklabels(lbls, rotation=30)
    else:
        ax.set_xticklabels(lbls)


def plotManifoooold(x,y,z,colorBy, ax):
    # make smoooth
    smooth = 12
    x = gaussian_filter1d(x, smooth)
    y = gaussian_filter1d(y, smooth)
    z = gaussian_filter1d(z, smooth)
    
    multicolor(ax,x,y,z,colorBy,c= transientcmap, threedim = True, etho = False, cg = 1)
    ax.scatter3D(x[::12], y[::12], z[::12], c=colorBy[::12], cmap=transientcmap, s=10)
    #ax.view_init(elev=40, azim=-15)
    ax.dist = 10
    axmin, axmax = -5, 5
    ticks = [axmin,0, axmax]
    
#    ax.set_xlim([axmin, axmax])
#    ax.set_ylim([axmin, axmax])
#    ax.set_zlim([axmin, axmax])
#    #
    ax.tick_params(axis='both', which='major', pad=0)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
#=============================================================================#
#                           Animating worms
#=============================================================================#
def width(x):
        """empirical worm width...i.e., I eyeballed a function."""
        a,b,x0 = 5, 10., 100
        return  35*((1 / (1 + np.exp(-x/a)))*(1 - 1 / (1 + np.exp(-(x-x0)/b)))-0.5)
    
def createWorm(x, y):
    """creates vertices for a worm from centerline points x1, y1. """
    lwidths = width(np.linspace(0,100, len(x)))
    # create orthogonal vectors
    e1 = np.vstack([np.diff(y), -np.diff(x)])
    e1 /= np.linalg.norm(e1, axis =0)
    e1 = np.pad(e1, ((0,0),(0,1)), 'constant')
    a = np.vstack([x,y])+lwidths*e1
    b = np.vstack([x,y])-lwidths*e1
    return np.concatenate([a, b[:,::-1]], axis=1).T

#=============================================================================#
#                           utility
#=============================================================================#
def sortnbin(x, y, nBins, rng):
    """takes a scatter plot and bins by some number of bins and a range."""
    
    # sort x and y
    y = y[np.argsort(x)]
    x = np.sort(x)
    # create bins
    _, b = np.histogram([],nBins,rng)
    c = np.digitize(x, b)
    
    std = []
    avg = []
    n = []
    for i in range(nBins):
        _t = y[c == i]
        std.append(np.std(_t))
        avg.append(np.mean(_t))
        n.append(len(_t))

    xPlot = b[:-1] + np.diff(b) *0.5
    return xPlot, np.array(avg), np.array(std)#/np.sqrt(n)
