# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:50:38 2017
plot assistant. make pretty plots.
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn import linear_model
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import explained_variance_score, r2_score
from scipy.ndimage.filters import gaussian_filter
from sklearn.preprocessing import StandardScaler
#custom
import dataHandler as dh
# change axes
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
mpl.rcParams["axes.labelsize"]=  10
mpl.rcParams["xtick.labelsize"]=  10
mpl.rcParams["ytick.labelsize"]=  10
mpl.rc('font', **{'sans-serif' : 'FiraSans','family' : 'sans-serif'})
mpl.rc('text.latex', preamble='\usepackage{sfmath}')
plt.rcParams['image.cmap'] = 'viridis'

#=============================================================================#
#                           Define UC colors
#=============================================================================#
UCyellow = ['#FFA319','#FFB547','#CC8214']
UCorange = ['#C16622','#D49464','#874718']
UCred    = ['#8F3931','#B1746F','#642822']
UCgreen  = ['#8A9045','#ADB17D','#616530','#58593F','#8A8B79','#3E3E23']
UCblue   = ['#155F83','#5B8FA8','#0F425C']
UCviolet = ['#350E20','#725663']
UCgray   = ['#767676','#D6D6CE']

UCmain   = '#800000'
#=============================================================================#
#
# cyclic colormap and other fun stuff
#
#=============================================================================#
cmap = {name:plt.get_cmap(name) for name in ('viridis', 'viridis_r')}
N = 50
levels = np.concatenate([np.linspace(0, np.pi, N, endpoint=False),
                         np.linspace(np.pi, 0, N+1, endpoint=True)])  # 2
colors = np.concatenate([cmap[name](np.linspace(0, 1, N)) 
                         for name in ('viridis', 'viridis_r')])           # 3

cyclon, _ = mpl.colors.from_levels_and_colors(levels, colors)

# continous behavior colors - training
colorBeh = {'AngleVelocity':'#DC143C', 'Eigenworm3':'#4876FF', 'Eigenworm2':'#4caf50', 'CMSVelocity':'#555555'}
# continous behavior colors - prediction
colorPred = {'AngleVelocity':'#6e0a1e', 'Eigenworm3':'#1c2f66', 'Eigenworm2':'#265728', 'CMSVelocity':'#333333'}
# discrete behaviors
#colDict = {-1:'red',0:'k',1:'green',2:'blue'}
# same colors as manifolds
colDict = {-1:'#C21807', 0: UCgray[1], 1:'#4AA02C', 2:'#0F52BA'}
labelDict = {-1:'Reverse',0:'Pause',1:'Forward',2:'Turn'}
# color the ethogram
ethocmap = mpl.colors.ListedColormap([mpl.colors.to_rgb('#C21807'), UCgray[1], mpl.colors.to_rgb('#4AA02C'), mpl.colors.to_rgb('#0F52BA')], name='etho', N=None)
ethobounds=[-1,0,1,2, 3]
ethonorm = mpl.colors.BoundaryNorm(ethobounds, ethocmap.N)


# rename behaviors for plots
names = {'AngleVelocity': 'Wave velocity',
         'Eigenworm3': 'Turns', 
         'Eigenworm2': 'Head swing',
         'Eigenworm1': 'Head swing'
        }

#=============================================================================#
#                           Histograms 1D and 2D
#=============================================================================#
def hist2d(x,y,nBinsX,nBinsY,rngX=None,rngY=None):
    if rngX == None and rngY == None:
        h2d, xp, yp = np.histogram2d(y,x,bins=(nBinsY,nBinsX), normed = True)
    else:
        h2d, xp, yp = np.histogram2d(y,x,bins=(nBinsY,nBinsX),range=[rngY,rngX], normed = True)
    extent = [yp[0],yp[-1],xp[0],xp[-1]]
    return h2d, extent
    
def histogram(data, bins, normed=False):
    """simple numpy hist wrapper."""
    hist, bins = np.histogram(data, bins, normed=normed)
    x = bins[:-1]+0.5*(bins[1]-bins[0])
    return x, hist

#=============================================================================#
#                           Animating worms
#=============================================================================#
def make_animation(fig, ax, data1, data2, frames):
    """use pythons built-in animation tools to make a centerline animation."""
    x1,y1 = data1[0].T
    x2,y2 = data2[0].T
    line1, = ax.plot(x1,y1, lw=20, color = '0.5')
    line2, = ax.plot(x2, y2, lw=20, color='r')
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1,line2
    # animation function.  This is called sequentially
    def animate(i, data1, data2= None):
        x1,y1 = data1[i].T
        x2,y2 = data2[i].T
        
        line1.set_data(x1, y1)
        line2.set_data(x2, y2)
        return line1, line2
        
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,fargs=[data1, data2],
                               frames=frames, interval=200, blit=True)
    plt.show()


def make_animation2(fig, ax, data1, data2, frames, color = 'gray'):
    """use pythons built-in animation tools to make a centerline animation."""
    def width(x):
        """empirical worm width...i.e., I eyeballed a function."""
        a,b,x0 = 10., 20., 90
        return  15*((1 / (1 + np.exp(-x/a)))*(1- 1 / (1 + np.exp(-(x-x0)/b))))
        
    
    def init():
        points = data1[0].reshape(-1, 1, 2)
        segments = np.hstack([points[:-1], points[1:]])
        lwidths = width(np.linspace(0,100, len(segments)))
        lc1 = LineCollection(segments, linewidths=lwidths,color=color)
        ax.add_collection(lc1)
        if data2 is not None:
            points2 = data2[0].reshape(-1, 1, 2)
            segments2 = np.hstack([points2[:-1], points2[1:]])
            
            lc2 = LineCollection(segments2, linewidths=lwidths,color=color)
            ax.add_collection(lc2)
            return lc1, lc2,
        return lc1
    lc1, lc2 = init()
    # animation function.  This is called sequentially
    def animate(i, data1, data2= None):
        points = data1[i].reshape(-1, 1, 2)
        segments = np.hstack([points[:-1], points[1:]])
        lc1.set_segments(segments)
        if data2 is not None:
            points2 = data2[i].reshape(-1, 1, 2)
            segments2 = np.hstack([points2[:-1], points2[1:]])
            lc2.set_segments(segments2)
            return lc1, lc2
        #line2.set_data(x2, y2)
        return lc1,# line2
        
    
    anim = animation.FuncAnimation(fig, animate, fargs=[data1, data2],
                               frames=frames, interval=200, blit=True)
    plt.show()
    
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
        
def make_animation3(fig, ax, data1, data2, frames, color = 'gray', save= False):
    """use pythons built-in animation tools to make a centerline animation."""
    if save:
        writer = animation.FFMpegWriter()
        #writer = Writer(fps=6, metadata=dict(artist='Me'), bitrate=1800)

        
    def init():
        x1,y1 = data1[0].T
        
        Vertices1 = createWorm(x1, y1)

        p1.set_xy(Vertices1)
        patch1=ax.add_patch(p1)
        if data2 is not None:
            x2,y2 = data2[0].T
            Vertices2 = createWorm(x2, y2)

            p2.set_xy(Vertices2)
            patch2 = ax.add_patch(p2)
            
            return patch1, patch2
        return patch1
        
    p1 = mpl.patches.Polygon(np.zeros((2,2)), closed=True, fc=color, ec='none')
    p2 = mpl.patches.Polygon(np.zeros((2,2)), closed=True, fc=color, ec='none')
    patch1, patch2= init()
    
    # animation function.  This is called sequentially
    def animate(i, data1, data2= None):
        x1,y1 = data1[i].T
        
        Vertices = createWorm(x1, y1)

        p1.set_xy(Vertices)
        patch1=ax.add_patch(p1)
        if data2 is not None:
            x2,y2 = data2[i].T
            Vertices2 = createWorm(x2, y2)
            p2.set_xy(Vertices2)
            patch2=ax.add_patch(p2)

            return patch1, patch2
        return patch1
        
    anim = animation.FuncAnimation(fig, animate, fargs=[data1, data2],
                               frames=frames, interval=166, blit=True, repeat=False)
    if save:
        anim.save('im2.mp4', writer=writer)
    plt.show()
    


def plot2DProjections(xS,yS, zS, fig, gsobj, colors = ['r', 'b', 'orange']):
    '''plot 3 projections into 2d for 3dim data sets. Takes an outer gridspec object to place plots.'''
    s, a = 0.05, 0.25
    inner_grid = gridspec.GridSpecFromSubplotSpec(1, 3,
    subplot_spec=gsobj, hspace=0.25, wspace=0.5)
    ax1 = plt.Subplot(fig, inner_grid[0])
    fig.add_subplot(ax1)
    ax1.scatter(xS,yS,color=colors[0], s=s, alpha = a)
    ax1.set_ylabel('Y')
    ax1.set_xlabel('X')
    
    ax2 = plt.Subplot(fig, inner_grid[1])
    fig.add_subplot(ax2)
    ax2.scatter(xS,zS,color=colors[1], s= s, alpha = a)
    ax2.set_ylabel('Z')
    ax2.set_xlabel('X')
    
    ax3 = plt.Subplot(fig, inner_grid[2])
    fig.add_subplot(ax3)
    ax3.scatter(zS,yS,color=colors[2], s=s, alpha = a)
    ax3.set_ylabel('Y')
    ax3.set_xlabel('Z')
    return [ax1, ax2, ax3]

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

def circle_scatter(axes, x_array, y_array, radius=0.5, **kwargs):
    """make scatter plot with axis unit radius.(behaves nice when zooming in)"""
    for x, y in zip(x_array, y_array):
        if np.isfinite(x):
            circle = plt.Circle((x,y), radius=radius, **kwargs)
            axes.add_patch(circle)
    return True

def plotHeatmap(T, Y, ax = None, vmin=-2, vmax=2):
    """nice looking heatmap for neural dynamics."""
    if ax is None:
        ax = plt.gca()
    cax1 = ax.imshow(Y, aspect='auto', interpolation='none', origin='lower',extent=[T[0],T[-1],len(Y),0],vmax=vmax, vmin=vmin)
    ax.set_xlabel('Time (s)')
    ax.set_yticks(np.arange(0, len(Y),25))
    ax.set_ylabel("Neuron")
    return cax1
    
def plotEigenworms(T, E, label, color = 'k'):
    """make an eigenworm plot"""
    plt.plot(T, E, color = color, lw=1)
    plt.ylabel(label)
    plt.xlabel('Time (s)')
    plt.xlim([0, np.max(T)])

def plotEthogram(ax, T, etho, alpha = 0.5, yValMax=1, yValMin=0, legend=0):
    """make a block graph ethogram for elegans behavior"""
    #colDict = {-1:'red',0:'k',1:'green',2:'blue'}
    labelDict = {-1:'Reverse',0:'Pause',1:'Forward',2:'Turn'}
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
        plt.legend(ncol=2)
    
def plotExampleCenterlines(dataSets, keyList, folder)  :
    """plot a few centerlines from different behaviors."""
    print 'plot centerline'
    nWorms = len(keyList)
    fig = plt.figure('Centerlines',(10, nWorms*3.4))
    gs = gridspec.GridSpec(nWorms, 10)
    for dindex, key in enumerate(keyList):
        cl, wc = dh.loadCenterlines(folder.format(key))
        
        for i, index in enumerate(np.arange(0,len(cl), int(len(cl)/9))):
            ax2 = plt.subplot(gs[dindex, i])
            x, y = cl[index][:,0], cl[index][:,1]
            x -= np.mean(x)
            y -= np.mean(y)
            ax2.plot(x,y)
            span = np.max([np.max(x)-np.min(x), np.max(y)-np.min(y)])
            ax2.set_xlim([np.min(x)-10, np.min(x)+span])
            ax2.set_ylim([np.min(y)-10, np.min(y)+span])
    
def plotBehaviorAverages(dataSets, keyList)  :
    """plot the mean calcium signal given a behavior."""
    print 'plot BTA'
    nWorms = len(keyList)
    fig = plt.figure('BehaviorAverage',(7, nWorms*3.4))
    gs = gridspec.GridSpec(4, nWorms)
    #gs = gridspec.GridSpec(nWorms, 1)
    for dindex, key in enumerate(keyList):
        data = dataSets[key]['Behavior']['Ethogram']
        Y = dataSets[key]['Neurons']['Activity'].T
        orderFwd = np.argsort(np.std(Y, axis=0))
        for index, bi in enumerate([1,-1, 2, 0]):
            indices = np.where(data==bi)[0]
            Ynew = Y[indices]
            m = np.mean(Ynew, axis=0)
            s = np.std(Ynew, axis=0)
            order = np.argsort(m)
            low, high = np.percentile(Ynew,[10,90], axis=0)
            #    orderFwd = order
            ax = plt.subplot(gs[index, dindex])

            plotHeatmap(np.arange(len(m)), np.mean(Ynew[:,orderFwd], axis=0)[np.newaxis,:], ax = ax, vmax=1, vmin=-1)
            ax.step(np.arange(len(m)), m[orderFwd],where='mid', alpha=1, color=colDict[bi])

            ax.set_xlabel('neurons (sorted by mean)')
            ax.set_ylabel(labelDict[bi])

            
def plotBehaviorOrderedNeurons(dataSets, keyList, behaviors):
    """plot the neural data as ordered by behaviors."""
    print 'plot neurons ordered by behavior'
    nWorms = len(keyList)
    fig = plt.figure('BehaviorOrdered Neurons',(12, nWorms*3.4))
    gs = gridspec.GridSpec(nWorms,1)
    #gs = gridspec.GridSpec(nWorms, 1)
    for dindex, key in enumerate(keyList):
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, len(behaviors),
            subplot_spec=gs[dindex], hspace=0.5, wspace=0.35, height_ratios=[0.5,1])
        for bindex, beh in enumerate(behaviors):
            x = dataSets[key]['Behavior'][beh]
            Y = dataSets[key]['Neurons']['Activity']
            xOrder = np.argsort(x)
            #plot sorted behavior
            ax = plt.Subplot(fig, inner_grid[0, bindex])
            ax.plot(x[xOrder], color=colorBeh[beh])
            ax.set_xlim([0, len(xOrder)])
            ax.set_ylabel(beh)
            fig.add_subplot(ax)
            # find interesting locations:
            ax.axvline(np.where(x[xOrder]>0)[0][0], color='k', lw=1, linestyle='--')
#            if beh == 'AngleVelocity':
#                ax.axvline(np.where(x[xOrder]>0)[0][0])
#            if beh=='Eigenworm3':
#                ax.axvline(np.where(np.sort(x)<-10)[0][-1])
#                ax.axvline(np.where(np.sort(x)>10)[0][0])
            ax.set_xticks([])
            #plot neural signal sorted
            ax2 = plt.Subplot(fig, inner_grid[1, bindex], sharex=ax)
            plotHeatmap(np.arange(len(Y[0])), gaussian_filter(Y[:,xOrder], (1,5)), ax =ax2,vmin=-0.5, vmax=1)
            ax2.set_xlabel('Neural activity ordered by behavior')
            # find interesting locations:
            ax2.axvline(np.where(x[xOrder]>0)[0][0], color='w', lw=1)
#            if beh == 'AngleVelocity':
#                ax2.axvline(np.where(x[xOrder]>0)[0][0], color='w', lw=0.5)
#            if beh=='Eigenworm3':
#                ax2.axvline(np.where(np.sort(x)<-10)[0][-1], color='w', lw=0.5)
#                ax2.axvline(np.where(np.sort(x)>10)[0][0], color='w', lw=0.5)
            #
            fig.add_subplot(ax2)
    gs.tight_layout(fig)
            
def plotBehaviorNeuronCorrs(dataSets, keyList, behaviors):
    """plot the neural data as ordered by behaviors."""
    print 'plot neurons behavior correlations.'
    #gs = gridspec.GridSpec(nWorms, 1)
    for dindex, key in enumerate(keyList):
        fig = plt.figure('Behavior correlates Neurons {}'.format(dindex),(12, 12))
        gs = gridspec.GridSpec(2, 1)
        
        Y = dataSets[key]['Neurons']['Activity']
        nNeur = len(Y)
        nRows = int(np.sqrt(nNeur))+1
        r2s = []
        scaler = StandardScaler()
        for bindex, beh in enumerate(behaviors):
            inner_grid = gridspec.GridSpecFromSubplotSpec(nRows,nRows,\
            subplot_spec=gs[bindex], hspace=0.05, wspace=0.05)
            x = dataSets[key]['Behavior'][beh]
            x = (x-np.mean(x))/np.std(x)
            
            for n in range(nNeur):
                ax = plt.subplot(inner_grid[int(n/nRows), n%nRows])
                r2 = np.corrcoef(x, Y[n])[0,1]**2
                r2s.append(r2)
                ax.scatter(x,Y[n], color=colorBeh[beh], s=0.25, alpha=0.1, label="{:.2f}".format(r2))
                
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.legend()
        r2scores = np.reshape(np.array(r2s), (-1, len(behaviors)))
        gs.tight_layout(fig)
        plt.figure()
        plt.subplot(231)
        plt.plot(r2scores[:,0], colorPred['AngleVelocity'], label = names['AngleVelocity'])
        if len(behaviors) >1:        
            plt.plot(r2scores[:,1], colorPred['Eigenworm3'], label = names['Eigenworm3'])
        plt.xlabel('Neurons')
        plt.ylabel(r'R^2 score')
        plt.subplot(232)
        plt.xlabel('R^2 scores')
        plt.ylabel(r'Distribution of R^2 scores')
        plt.hist(r2scores[:,0], bins = 10,color = colorPred['AngleVelocity'])
        plt.subplot(233)
        plt.xlabel('R^2 scores')
        plt.ylabel(r'Distribution of R^2 scores')
        if len(behaviors) >1:   
            plt.hist(r2scores[:,1], bins = 10,color = colorPred['Eigenworm3'])
        
        plt.subplot(234)
        plt.plot(dataSets[key]['Behavior']['AngleVelocity']+5, ':', color = colorBeh['AngleVelocity'])
        plt.plot(Y[np.argsort(r2scores[:,0])[-10:]].T, color = colorPred['AngleVelocity'], alpha=0.8)
        plt.subplot(235)
        if len(behaviors) >1:   
            plt.plot(dataSets[key]['Behavior']['Eigenworm3']+10, ':' ,color = colorBeh['Eigenworm3'])
            plt.plot(Y[np.argsort(r2scores[:,1])[-10:]].T, color = colorPred['Eigenworm3'], alpha=0.8)
        plt.show()
            #ax.fill_between(range(len(m)), m-s, m+s, alpha=0.5)
###############################################    
# 
# velocity versus angle veocity10
#
##############################################       
def plotVelocityTurns(dataSets, keyList):
    """plot velocity in relation to CMS velocity."""
    nWorms = len(keyList)
    fig = plt.figure('Eigenworms after Projection',(10, nWorms*3.4))
    gs = gridspec.GridSpec(nWorms*2, 1)
    for dindex, key in enumerate(keyList):
        
        data = dataSets[key]
        ax = plt.subplot(gs[2*dindex])
        ax.set_title(key)
        vel = np.copy(data['Behavior']['CMSVelocity'])
        vel = (vel-np.min(vel))
        vel /= np.max(vel)
        
        vel *= np.max(data['Behavior']['AngleVelocity'])
        vel += np.min(data['Behavior']['AngleVelocity'])
        yMin, yMax = np.min(data['Behavior']['AngleVelocity']), np.max(data['Behavior']['AngleVelocity'])
        plotEthogram(ax, data['Neurons']['Time'],  data['Behavior']['Ethogram'], alpha = 0.25, yValMin=yMin,yValMax=yMax )        
        plotEigenworms(data['Neurons']['Time'], data['Behavior']['AngleVelocity'], color = colorBeh['AngleVelocity'],label = names['AngleVelocity'])
        plotEigenworms(data['Neurons']['Time'], vel, color = 'k',label = 'CMS velocity')
        plt.legend()
       
        ax2 = plt.subplot(gs[2*dindex+1])
        yMin, yMax = np.nanmin(data['Behavior']['Eigenworm3']), np.nanmax(data['Behavior']['Eigenworm3'])
        plotEthogram(ax2, data['Neurons']['Time'],  data['Behavior']['Ethogram'], alpha = 0.25,yValMin=yMin,yValMax=yMax)
        plotEigenworms(data['Neurons']['Time'], data['Behavior']['Eigenworm3'],color = colorBeh['Eigenworm3'], label = names['Eigenworm3'])
        #plot cms motion        
    
    plt.tight_layout()
    plt.show()  

###############################################    
# 
# plot neural data as line plots
#
############################################## 
def neuralActivity(dataSets, keyList):
    nWorms = len(keyList)
    fig = plt.figure('Lines',(10, nWorms*3.4))
    gs = gridspec.GridSpec(nWorms, 1)
    
    for dindex, key in enumerate(keyList):
        print 'Plotting lines ', key
        
        data = dataSets[key]
        time = data['Neurons']['Time']
        ax = plt.subplot(gs[dindex])
        currPos = 0
        for line in data['Neurons']['Activity']:
            plt.plot(time, line+currPos, 'k-', lw=1)
            currPos += np.max(line)*1.1
        plt.ylabel('Neural Activity')
        plt.xlabel('Time (s)')
    gs.tight_layout(fig)
    plt.show()
###############################################    
# 
# full figures
#
##############################################    
def plotDataOverview(dataSets, keyList):
    """plot ethogram and heatmap"""
    nWorms = len(keyList)
    fig = plt.figure('Overview',(10, nWorms*3.4))
    gs = gridspec.GridSpec(nWorms*2, 4,
                           width_ratios=[1,0.1, 2, 1])
    for dindex, key in enumerate(keyList):
        print 'Plotting overview of ', key
        
        data = dataSets[key]
        ax = plt.subplot(gs[2*dindex:2*dindex+2,0])
        ax.set_title(key)
        hm = plotHeatmap(data['Neurons']['Time'], data['Neurons']['Activity'])
        ax2 = plt.subplot(gs[2*dindex:2*dindex+2,1])
        plt.colorbar(hm, cax=ax2)
             
        ax3 = plt.subplot(gs[2*dindex,2])
        yMin, yMax = np.min(data['Behavior']['AngleVelocity']), np.max(data['Behavior']['AngleVelocity'])
        #print len(data['Neurons']['Time']),  len(data['Behavior']['Ethogram'])
        plotEthogram(ax3, data['Neurons']['Time'],  data['Behavior']['Ethogram'], alpha = 0.25, yValMin=yMin,yValMax=yMax )        
        plotEigenworms(data['Neurons']['Time'], data['Behavior']['AngleVelocity'], color = colorBeh['AngleVelocity'],label = names['AngleVelocity'])
        # plot cms velocity
        
        
        ax4 = plt.subplot(gs[2*dindex+1,2])
        yMin, yMax = np.nanmin(data['Behavior']['Eigenworm3']), np.nanmax(data['Behavior']['Eigenworm3'])
        plotEthogram(ax4, data['Neurons']['Time'],  data['Behavior']['Ethogram'], alpha = 0.25,yValMin=yMin,yValMax=yMax)
        plotEigenworms(data['Neurons']['Time'], data['Behavior']['Eigenworm3'],color = colorBeh['Eigenworm3'], label = names['Eigenworm3'])
        #plot cms motion        
        ax5 = plt.subplot(gs[2*dindex:2*dindex+2,3])
        ax5.plot(data['Behavior']['X'], data['Behavior']['Y'])
    plt.tight_layout()
    plt.show()
###############################################    
# 
# full figures
#
##############################################    
def plotDataOverview2(dataSets, keyList, resultDict):
    """plot ethogram and heatmap plus behaviors"""
    nWorms = len(keyList)
    fig = plt.figure('Overview',(10, nWorms*6.8))
    gs = gridspec.GridSpec(nWorms*4,2, width_ratios=[1,0.1], height_ratios=np.tile([2,0.5,1,1],nWorms))
    for dindex, key in enumerate(keyList):
        print 'Plotting overview of ', key
        order =  resultDict[key]['PCA']['neuronOrderPCA']
        data = dataSets[key]
        ax = plt.subplot(gs[4*dindex, 0])
        ax.set_title(key)
        hm = plotHeatmap(data['Neurons']['Time'], data['Neurons']['Activity'][order])
        ax2 = plt.subplot(gs[4*dindex,1])
        plt.colorbar(hm, cax=ax2)
             
        
        ax3 = plt.subplot(gs[4*dindex+1,0])
        #yMin, yMax = np.min(data['Behavior']['AngleVelocity']), np.max(data['Behavior']['AngleVelocity'])
        #print len(data['Neurons']['Time']),  len(data['Behavior']['Ethogram'])
        plotEthogram(ax3, data['Neurons']['Time'],  data['Behavior']['Ethogram'], alpha = 1, yValMin=0,yValMax=1 )        
        #plotEigenworms(data['Neurons']['Time'], data['Behavior']['AngleVelocity'], color = colorBeh['AngleVelocity'],label = names['AngleVelocity'])
        
        ax4 = plt.subplot(gs[4*dindex+2, 0])
        plotEigenworms(data['Neurons']['Time'], data['Behavior']['AngleVelocity'], color = colorBeh['AngleVelocity'],label = names['AngleVelocity'])
        
        #yMin, yMax = np.nanmin(data['Behavior']['Eigenworm3']), np.nanmax(data['Behavior']['Eigenworm3'])
        #plotEthogram(ax4, data['Neurons']['Time'],  data['Behavior']['Ethogram'], alpha = 0.25,yValMin=yMin,yValMax=yMax)
        #plotEigenworms(data['Neurons']['Time'], data['Behavior']['Eigenworm3'],color = colorBeh['Eigenworm3'], label = names['Eigenworm3'])
        
        ax5 = plt.subplot(gs[4*dindex+3, 0])
        yMin, yMax = np.nanmin(data['Behavior']['Eigenworm3']), np.nanmax(data['Behavior']['Eigenworm3'])
        #plotEthogram(ax5, data['Neurons']['Time'],  data['Behavior']['Ethogram'], alpha = 0.25,yValMin=yMin,yValMax=yMax)
        plotEigenworms(data['Neurons']['Time'], data['Behavior']['Eigenworm3'],color = colorBeh['Eigenworm3'], label = names['Eigenworm3'])
        
    plt.tight_layout()
    plt.show()

    
def plotNeurons3D(dataSets, keyList, threed = True):
    """plot neuron locations."""
    nWorms = len(keyList)
    fig = plt.figure('Neurons',(6.8, nWorms*3.4))
    gs = gridspec.GridSpec(nWorms,1, hspace=0.25, wspace=0.25)
    for dindex, key in enumerate(keyList):
        data = dataSets[key]
#        pca = PCA(n_components = 3)
#        pca.fit_transform(data['Neurons']['Positions'])
#        xS, yS, zS = pca.components_
        xS, yS, zS = data['Neurons']['Positions']
        if threed:
            # make 3d scatter of neurons
            ax = plt.subplot(gs[dindex], projection='3d')
            ax.scatter(xS,yS,zS, c = 'r', s=10)
            ax.set_xlim(np.min(xS),np.max(xS))
            ax.set_ylim(np.min(yS),np.max(yS))
            ax.set_zlim(np.min(zS),np.max(zS))
        else:
            # plot projections of neurons
            inner_grid = gridspec.GridSpecFromSubplotSpec(1, 3,
            subplot_spec=gs[dindex], hspace=0.25, wspace=0.5)
            ax1 = plt.Subplot(fig, inner_grid[0])
            fig.add_subplot(ax1)
            ax1.scatter(xS,yS,color=UCred[0])
            ax1.set_ylabel('Y')
            ax1.set_xlabel('X')
            
            ax2 = plt.Subplot(fig, inner_grid[1])
            fig.add_subplot(ax2)
            ax2.scatter(xS,zS,color=UCblue[0])
            ax2.set_ylabel('Z')
            ax2.set_xlabel('X')
            
            ax3 = plt.Subplot(fig, inner_grid[2])
            fig.add_subplot(ax3)
            ax3.scatter(zS,yS,color=UCorange[1])
            ax3.set_ylabel('Y')
            ax3.set_xlabel('Z')
    
###############################################    
# 
# neuronal signal pca or ica, first n components and weights plotted
#
############################################## 
def singlePCAResult(fig, gridloc, Neuro, results, time, flag):
    """plot PCA of one dataset"""
    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 3,
        subplot_spec=gridloc, hspace=0.5, wspace=0.35, width_ratios=[2,0.2, 1])
    
    ax1 = plt.Subplot(fig, inner_grid[0, 0])
    # plot neurons ordered by weight in first PCA component
    cax1 = plotHeatmap(time,  Neuro[results['neuronOrderPCA']], ax= ax1)
    ax1.set_xlabel('Time (s)')
    fig.add_subplot(ax1)        
    axcb = plt.Subplot(fig, inner_grid[0, 1])        
    cbar = plt.colorbar(cax1, cax=axcb, use_gridspec = True)
    cbar.set_ticks([-2,0,2])
    cbar.set_ticklabels(['<-2',0,'>2'])
    fig.add_subplot(axcb) 
    
    # plot the weights
    ax2 = plt.Subplot(fig, inner_grid[0,2])
    pcs = results['neuronWeights']
    # normalize by max for each group
    #pcs = np.divide(pcs.T,np.max(np.abs(pcs), axis=1)).T
    rank = np.arange(0, len(pcs))
    for i in range(np.min([3,pcs.shape[1]])):
        y= pcs[:,i]
        # normalize
        #y-=np.min(y)
        #y /=np.max(y)
        ax2.fill_betweenx(rank, np.zeros(len(Neuro)),y[results['neuronOrderPCA']], step='pre', alpha=1.0-i*0.2)
    #ax2.fill_betweenx(rank, np.zeros(len(Neuro)),pcs[:,1][results['neuronOrderPCA']], step='pre')       
    #ax2.fill_betweenx(rank, np.zeros(len(Neuro)),pcs[:,2][results['neuronOrderPCA']], step='pre')       
    ax2.set_xlabel('Neuron weight')
    ax2.spines['left'].set_visible(False)
    ax2.set_yticks([])
    fig.add_subplot(ax2)
    
    ax3 = plt.Subplot(fig, inner_grid[1,0])
    for i in range(np.min([len(results['pcaComponents']), 3])):
        y = results['pcaComponents'][i]
        # normalize
        y =y -np.min(y)
        y =y/np.max(y)
        ax3.plot(time, i+y, label='Component {}'.format(i+1), lw=0.5)
    #ax3.legend()
    ax3.set_ylabel('Neural activity components')
    ax3.set_xlabel('Time (s)')
    ax3.set_xlim([np.min(time), np.max(time)])
    fig.add_subplot(ax3)
    ax4 = plt.Subplot(fig, inner_grid[1,2])
    nComp = results['nComp']
    
    if flag=='SVM':
        ax4.set_ylabel('F1 score')
        ax4.step(np.arange(nComp),results['expVariance'], where = 'pre')
    else:
        ax4.fill_between(np.arange(0.5,nComp+0.5),results['expVariance']*100, step='post', color='k', alpha=0.75)
        #ax4.step(np.arange(1,nComp+1),np.cumsum(results['expVariance'])*100, where = 'pre')
        ax4.plot(np.arange(1,nComp+1),np.cumsum(results['expVariance'])*100, 'ko-', lw=1)
        ax4.set_ylabel('Explained variance (%)')
        ax4.set_yticks([0,25,50,75,100])
    ax4.set_xlabel('Number of components')
    fig.add_subplot(ax4)
    
def plotPCAresults(dataSets, resultSet, keyList, pars, flag = 'PCA',testset=None ):
    """make an overview figure with PCA weights and components."""
    nWorms = len(keyList)
    fig = plt.figure('{}'.format(flag),(6.8, nWorms*3.4))
    outer_grid = gridspec.GridSpec(nWorms, 1, hspace=0.25, wspace=0.25)
    
    for kindex, key in enumerate(keyList):
        data = dataSets[key]
        if pars['useRank']:
            Neuro = data['Neurons']['rankActivity']
        else:
            Neuro = data['Neurons']['Activity']
        time = data['Neurons']['Time']
        if testset is not None:
            Neuro = Neuro[:,testset]
            time = data['Neurons']['Time'][testset]
        results = resultSet[key][flag]
       
        gridloc=outer_grid[kindex]
        singlePCAResult(fig, gridloc, Neuro, results, time, flag)
        
    outer_grid.tight_layout(fig)        
    
    
###############################################    
# 
# neuronal signal pca, plot in 3D with behavior labels
#
############################################## 
def plotPCAresults3D(dataSets, resultSet, keyList,pars,  col = 'phase', flag = 'PCA', smooth = 3, colorBy=None):
    """Show neural manifold with behavior label."""
    nWorms = len(keyList)
    print 'colored by ', col
    fig2 = plt.figure('{} projections'.format(flag),(6.8, nWorms*3.4))
    fig3 = plt.figure('{} temporal'.format(flag),(6.8, nWorms*3.4))
    fig1 = plt.figure('{} manifold'.format(flag),(6.8, nWorms*3.4))
    outer_grid2 = gridspec.GridSpec(nWorms, 3, hspace=0.25, wspace=0.25)
    outer_grid = gridspec.GridSpec(nWorms, 1, hspace=0.25, wspace=0.25)
    outer_grid1 = gridspec.GridSpec(nWorms, 1, hspace=0.25, wspace=0.25)
    
    
    for kindex, key in enumerate(keyList):
        data = dataSets[key]
        inner_grid1 = gridspec.GridSpecFromSubplotSpec(1, 2,
            subplot_spec=outer_grid1[kindex], hspace=0.5, wspace=0.35, width_ratios=[1,0.2])
        inner_grid3 = gridspec.GridSpecFromSubplotSpec(3, 1,
            subplot_spec=outer_grid[kindex], hspace=0.5, wspace=0.35)
        
        results = resultSet[key][flag]
        x,y,z = results['pcaComponents'][:3,]
        x = gaussian_filter1d(x, smooth)
        y = gaussian_filter1d(y, smooth)
        z = gaussian_filter1d(z, smooth)
        
        etho=False
        if colorBy is None:
            if col == 'phase':
                colorBy = np.arctan2(data['Behavior']['Eigenworm2'],data['Behavior']['Eigenworm1'])/np.pi
                cm = cyclon
            elif col == 'velocity':
                colorBy = np.copy(data['Behavior']['AngleVelocity'])
                colorBy -= np.mean(data['Behavior']['AngleVelocity'])
                colorBy /= np.std(data['Behavior']['AngleVelocity'])
                cm = 'jet'
            elif col =='turns':
                colorBy = np.copy(data['Behavior']['Eigenworm3'])
                colorBy -= np.mean(data['Behavior']['Eigenworm3'])
                colorBy /= np.std(data['Behavior']['Eigenworm3'])
                cm = 'jet'
            elif col=='time':
                colorBy = data['Neurons']['Time']
                cm = 'magma'
            else:
                colorBy = np.reshape(np.array(data['Behavior']['Ethogram']), (-1, ))
                cm = ethocmap
                etho = True
        else:
            # manually set the color by values given in col
            cm = ethocmap
            
        #etho[np.isnan(etho)] = 1
        #print ethocmap[0], ethocmap[1]
        #ax3 = plt.Subplot(fig, outer_grid[kindex])
        coarsegrain = 1
        ax4 = fig1.add_subplot(inner_grid1[0,0], projection = '3d')
        cax = multicolor(ax4,x,y,z,colorBy,c = cm, threedim = True, etho=etho,  cg =coarsegrain)
        ax4.set_ylabel('{} components 2'.format(flag))
        ax4.set_xlabel('{} components 1'.format(flag))
        ax4.set_zlabel('{} components 3'.format(flag))
        if etho:
            
            #axcb = plt.Subplot(fig1, inner_grid[0, 1])     
            axcb = fig1.add_subplot(inner_grid1[0, 1])
            cbar = plt.colorbar(cax, cax=axcb, use_gridspec = True, norm=ethonorm, drawedges=False)
            cbar.ax.get_yaxis().set_ticks([])
            for j, lab in enumerate(['Reverse','Pause','Forward','Turn']):
                cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center', color='white')
            cbar.ax.get_yaxis().labelpad = 15
            #fig1.add_subplot(axcb) 
        else:
            axcb = fig1.add_subplot(inner_grid1[0, 1])
            cbar= plt.colorbar(cax, cax = axcb, use_gridspec=True)
            cbar.ax.get_yaxis().labelpad = 15
        cbar.outline.set_visible(False)
        cbar.ax.set_ylabel(col, rotation=270)

                
        ax1= fig2.add_subplot(outer_grid2[kindex, 0])
       
        multicolor(ax1,x,y,z,colorBy,c = cm,threedim = False, etho=etho,  cg =coarsegrain)
        ax1.set_ylabel('Y')
        ax1.set_xlabel('X')

        ax2= fig2.add_subplot(outer_grid2[kindex, 1])
        multicolor(ax2,x,z,z,colorBy,c = cm,threedim = False, etho=etho,  cg =coarsegrain)
        ax2.set_ylabel('Z')
        ax2.set_xlabel('X')
        
        ax3= fig2.add_subplot(outer_grid2[kindex, 2])
        multicolor(ax3,z,y,x,colorBy,c = cm,threedim = False, etho=etho,  cg =coarsegrain)
        ax3.set_ylabel('Y')
        ax3.set_xlabel('Z')
        
        # temporal evolution
        ax4 = fig3.add_subplot(inner_grid3[0])
        multicolor(ax4,data['Neurons']['Time'],x,x,colorBy,c = cm,threedim = False, etho=etho,  cg =coarsegrain)
        ax4.set_ylabel('PCA 1')
        ax4 = fig3.add_subplot(inner_grid3[1])
        multicolor(ax4,data['Neurons']['Time'],y,x,colorBy,c = cm,threedim = False, etho=etho,  cg =coarsegrain)
        
        ax4.set_ylabel('PCA 2')
        ax4 = fig3.add_subplot(inner_grid3[2])
        multicolor(ax4,data['Neurons']['Time'],z,x,colorBy,c = cm,threedim = False, etho=etho,  cg =coarsegrain)
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('PCA 3')
    outer_grid2.tight_layout(fig2)
    outer_grid.tight_layout(fig3)
    outer_grid1.tight_layout(fig1)


def plotPCAcorrelates(dataSets, resultDict, keyList, pars, flag='PCA'):
    """correlate PCA with all sorts of stuffs"""
    nWorms = len(keyList)
    fig3 = plt.figure('{} Correlates'.format(flag),(3.4, 7))
    #outer_grid = gridspec.GridSpec(nWorms, 1, hspace=0.25, wspace=0.25)
    outer_grid = gridspec.GridSpec(1, 1, hspace=0.25, wspace=0.25)
    inner_grid = gridspec.GridSpecFromSubplotSpec(3, 1,
        subplot_spec=outer_grid[0], hspace=0.5, wspace=0.35)
    for kindex, key in enumerate(keyList):
        data = dataSets[key]
        
        x,y,z = resultDict[key][flag]['pcaComponents'][:3,]
        
        theta = np.unwrap(np.arctan2(y, z))
        #velo = dh.savitzky_golay(theta, window_size=17, order=5, deriv=1, rate=1)
        #velo = dh.savitzky_golay(x, window_size=7, order=3, deriv=2, rate=1)
        phase = np.arctan2(data['Behavior']['Eigenworm2'],data['Behavior']['Eigenworm1'])/np.pi
        corrs = [data['Behavior']['CMSVelocity']*100, data['Behavior']['AngleVelocity'], data['Behavior']['Eigenworm3'],  data['Behavior']['Eigenworm1'], data['Behavior']['Ethogram']]
        corrNames = ['CMS velocity', 'Phase velocity', 'Turns', 'Head angle', 'Ethogram']        
        corrNames = ['Phase velocity']     
        size, a = 2, 0.1
        for pcix, pc in enumerate([x,y,z]):
            for cix, correlate in enumerate(corrs[1:2]):
                ax = fig3.add_subplot(inner_grid[pcix, cix])
                # Train the model
                regr = linear_model.LinearRegression()
                regr.fit(pc.reshape(-1, 1), correlate.reshape(-1, 1))
                r2 = regr.score(pc.reshape(-1, 1), correlate.reshape(-1, 1))
                ax.scatter(pc, correlate, s=size, alpha=a, label= 'R2 = {:.2f}'.format(r2))
                ax.set_ylabel(corrNames[cix])
                ax.legend()
                

                
#            ax = fig3.add_subplot(inner_grid[pcix, 1])
#            ax.scatter(pc, data['Behavior']['AngleVelocity'], s=size, alpha=a)
#            ax = fig3.add_subplot(inner_grid[pcix, 2])
#            ax.scatter(pc, data['Behavior']['Eigenworm3'], s=size, alpha=a)
#            ax = fig3.add_subplot(inner_grid[pcix,3])
#            ax.scatter(pc, phase, s=size, alpha=a)
#            ax = fig3.add_subplot(inner_grid[pcix,4])
#            ax.scatter(np.diff(pc), data['Behavior']['Eigenworm3'][:-1], s=size, alpha=a)
    outer_grid.tight_layout(fig3)
        
###############################################    
# 
# plot LASSO and other linear models
#
##############################################  
def plotSingleLinearFit(fig, gridloc, pars, results, data, splits, behaviors):
    inner_grid = gridspec.GridSpecFromSubplotSpec(len(behaviors), 5,
                subplot_spec=gridloc, hspace=1, wspace=0.5, width_ratios=[3,1,1,1,1])
    for lindex, label in enumerate(behaviors):
        #weights, intercept, alpha, _,_ = resultSet[key][fitmethod][label]
        weights = results[label]['weights']
        intercept = results[label]['intercepts']
        if pars['useRank']:
            x = data['Neurons']['rankActivity']
        else:
            x = data['Neurons']['Activity']
        y = data['Behavior'][label]
        trainingsInd, testInd = splits[label]['Train'], splits[label]['Test']
    
        # calculate y from model
        yPred = np.dot(weights, x) + intercept
        
        yTrain = np.ones(yPred.shape)*np.nan
        yTrain[trainingsInd] = yPred[trainingsInd]
        
        yTest =  np.ones(yPred.shape)*np.nan
        yTest[testInd] = yPred[testInd]
        
        #if random=='random':
        #    yTest = yPred
        # plot training and test set behavior and prediction
        ax1 = plt.Subplot(fig, inner_grid[lindex, 0])
        ax1.axvspan(data['Neurons']['Time'][np.min(testInd)], data['Neurons']['Time'][np.max(testInd)], color=UCgray[1])
        ax1.text(data['Neurons']['Time'][int(np.mean(testInd))], np.max(y),'Predicted', horizontalalignment='center')
        #ax1.plot(data['Neurons']['Time'], yTrain, color=colorBeh[label], label = 'Training', alpha =0.4, lw=2)
        ax1.plot(data['Neurons']['Time'], y, color=colorBeh[label], label = 'Behavior', lw=1)
        ax1.plot(data['Neurons']['Time'], yTest, color=colorPred[label], label = r'$R^2$ {0:.2f}'.format(float(results[label]['scorepredicted'])), lw=1)
        ax1.set_xlim(np.percentile(data['Neurons']['Time'], [0,100]))    
        ax1.set_ylabel(names[label])
        if lindex==len(behaviors)-1:
            ax1.set_xlabel('Time (s)')
        
        ax1.legend(loc=(0.0,0.95), ncol = 2)
        fig.add_subplot(ax1)
        
        # show how predictive each additional neuron is
        ax4 = plt.Subplot(fig, inner_grid[lindex, 2])
        ax4.plot(np.arange(1,len(results[label]['cumulativeScore'])+1),results[label]['cumulativeScore'], color=colorPred[label],marker='o',  markerfacecolor="none",markersize=5)
        ax4.plot(np.arange(1,len(results[label]['cumulativeScore'])+1),results[label]['individualScore'], color=colorBeh[label],marker='o', markerfacecolor="none", markersize=5)
          
        ax4.set_ylabel(r'$R^2$ score')
        if lindex==len(behaviors)-1:
            ax4.set_xlabel('Number of neurons')
        fig.add_subplot(ax4)
        # plot prediction scatter
        ax5 = plt.Subplot(fig, inner_grid[lindex, 3])
        if lindex==len(behaviors)-1:
            ax5.set_xlabel('True behavior')
        ax5.set_ylabel('Predicted')
        ax5.scatter(y[testInd], yPred[testInd], alpha=0.05,s=5, color=colorPred[label])
        fig.add_subplot(ax5)
        # plot cumulative MSE of fits
        ax6 = plt.Subplot(fig, inner_grid[lindex, 4])
        if lindex==len(behaviors)-1:
            ax6.set_xlabel('Number of neuronsr')
        ax6.set_ylabel('MSE')
        ax6.plot(np.arange(1,len(results[label]['MSE'])+1),results[label]['MSE'], color=colorPred[label],marker='o',  markerfacecolor="none",markersize=5)
        fig.add_subplot(ax6)
        
    # plot weights
    ax3 = plt.Subplot(fig, inner_grid[:,1])
    for lindex, label in enumerate(behaviors):
        weights = results[label]['weights']
        
        if lindex == 0:
            indices = np.arange(len(x))
            indices = np.argsort(weights)
        rank = np.arange(0, len(weights))
        ax3.fill_betweenx(rank, np.zeros(len(weights)),weights[indices]/np.max(weights), step='pre', color=colorBeh[label], alpha = 0.5)
    
    ax3.set_ylabel('Neuron weights')
    ax3.spines['left'].set_visible(False)
    ax3.set_yticks([])
    fig.add_subplot(ax3)        
        
def plotLinearModelResults(dataSets, resultSet, keyList, pars, fitmethod='LASSO', behaviors = ['AngleVelocity', 'Eigenworm3'], random = 'none'):
    """make an overview figure with Lasso weights and components."""
    nWorms = len(keyList)
    
    fig = plt.figure(fitmethod,(2*6.8, nWorms*1.7*len(behaviors)))
    outer_grid = gridspec.GridSpec(nWorms, 1, hspace=0.25, wspace=0.25)
    
    for kindex, key in enumerate(keyList):
        gridloc = outer_grid[kindex]
        splits = resultSet[key]['Training']  
        data = dataSets[key]
        
        results = resultSet[key][fitmethod]
        plotSingleLinearFit(fig, gridloc, pars, results, data, splits, behaviors)
    outer_grid.tight_layout(fig)
###############################################     
# 
# plot residuals of linear model fit
#
##############################################    
def plotLinearModelResiduals(dataSets, resultSet, keyList, fitmethod='LASSO'):
    """show how much prediction changes when adding neurons."""
    nWorms = len(keyList)
    
    fig = plt.figure(fitmethod,(6.8, nWorms*3.4))
    outer_grid = gridspec.GridSpec(nWorms, 1, hspace=0.25, wspace=0.25)
    
    for kindex, key in enumerate(keyList):
        trainingsInd, testInd = resultSet[key]['Training']['Indices']  
        data = dataSets[key]
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2,
                subplot_spec=outer_grid[kindex], hspace=0.5, wspace=0.15, width_ratios=[3,1])
        for lindex, label in enumerate(['AngleVelocity', 'Eigenworm3']):
            #weights, intercept, alpha, _,_ = resultSet[key][fitmethod][label]
            weights = resultSet[key][fitmethod][label]['weights']
            intercept = resultSet[key][fitmethod][label]['intercepts']
            
            
            y = data['Behavior'][label]
            x = data['Neurons']['Activity']
            # calculate y from model
            yPred = np.dot(weights, x) + intercept
            # calculate relevant neurons
            
            weightsSortedIndex = np.argsort(np.abs(weights))
            weightsSorted = weights[weightsSortedIndex]
            Xrel = np.copy(x)[weightsSortedIndex]
            Xrel = Xrel[weightsSorted!=0,:]
            weightsRel = weightsSorted[weightsSorted!=0]
            ax1 = plt.Subplot(fig, inner_grid[lindex, 0])
            ax1.plot(data['Neurons']['Time'], y, color=colorBeh[label], label = 'Behavior', lw=1.5)
            for index in range(0,len(Xrel),3):
            # plot prediction for each neuron
                # rescale weights
                w = weightsRel[:index+1]
                yPred = np.dot(weightsRel[:index+1], Xrel[:index+1]) + intercept
                ax1.plot(data['Neurons']['Time'], yPred, color='k', alpha = 1./len(Xrel)*(index+1), label = 'Training', lw=1)
            # shade training areas
            #ax1.fill_between(data['Neurons']['Time'][trainingsInd], y1=np.zeros(len(trainingsInd)),\
            #y2=np.zeros(len(trainingsInd))+np.max(y*1.1))
            ax1.set_xlim(np.percentile(data['Neurons']['Time'], [0,100]))    
            ax1.set_ylabel(label)
            ax1.set_xlabel('Time (s)')
            fig.add_subplot(ax1)
        # plot weights
            
        ax3 = plt.Subplot(fig, inner_grid[:,1])
        for lindex, label in enumerate(['AngleVelocity', 'Eigenworm3']):
            weights = resultSet[key][fitmethod][label]['weights']
            
            if lindex == 0:
                indices = np.arange(len( data['Neurons']['Activity']))
                indices = np.argsort(weights)
            rank = np.arange(0, len(weights))
            ax3.fill_betweenx(rank, np.zeros(len(weights)),weights[indices]/np.max(weights), step='pre', color=colorBeh[label], alpha = 0.5)
        
        ax3.set_xlabel('Neuron weights ({})'.format(fitmethod))
        ax3.spines['left'].set_visible(False)
        ax3.set_yticks([])
        fig.add_subplot(ax3)
    outer_grid.tight_layout(fig)
        
###############################################    
# 
# plot weights on neuron distribution
#
##############################################  
def plotWeightLocations(dataSets, resultSet, keyList, fitmethod='ElasticNet'):
    """plot neuron locations."""
    nWorms = len(keyList)
    fig = plt.figure('Neurons',(6.8, nWorms*3.4))
    gs = gridspec.GridSpec(nWorms,1, hspace=0.25, wspace=0.25)
    for dindex, key in enumerate(keyList):
        data = dataSets[key]
        xS, yS, zS = data['Neurons']['Positions']
        # pca weight locations to align
        if fitmethod == 'PCA':
            weightsAV = resultSet[key][fitmethod]['neuronWeights'][:,0]
            weightsEW = resultSet[key][fitmethod]['neuronWeights'][:,1]
        else:
            
            # fitted weights
            weightsAV = resultSet[key][fitmethod]['AngleVelocity']['weights']
            weightsEW = resultSet[key][fitmethod]['Eigenworm3']['weights']
        # find non-zero weighted neurons
        indexAV = weightsAV != 0
        indexEW = weightsEW != 0
        # plot projections of neurons
        s0,s1,s2 = 64, 64, 32 # size of gray, red, blue neurons
        #
        s1 = np.abs(weightsAV[indexAV])/np.max(weightsAV)*64
        s2 = np.abs(weightsEW[indexEW])/np.max(weightsEW)*32
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 3,
        subplot_spec=gs[dindex], hspace=0.25, wspace=0.5)
        ax1 = plt.Subplot(fig, inner_grid[0])
        fig.add_subplot(ax1)
        ax1.scatter(xS,yS,color=UCgray[1], s = s0)
        #circle_scatter(ax1, xS, yS, radius=s0, color=UCgray[0])
        ax1.scatter(xS[indexAV],yS[indexAV],color=colorBeh['AngleVelocity'], s = s1)
        ax1.scatter(xS[indexEW],yS[indexEW],color=colorBeh['Eigenworm3'], s = s2)
        
        ax1.set_ylabel('Y')
        ax1.set_xlabel('X')
        
        ax2 = plt.Subplot(fig, inner_grid[1])
        fig.add_subplot(ax2)
        ax2.scatter(xS,zS,color=UCgray[1], s = s0)
        ax2.scatter(xS[indexAV],zS[indexAV],color=colorBeh['AngleVelocity'], s = s1)
        ax2.scatter(xS[indexEW],zS[indexEW],color=colorBeh['Eigenworm3'], s = s2)
        ax2.set_ylabel('Z')
        ax2.set_xlabel('X')
        
        ax3 = plt.Subplot(fig, inner_grid[2])
        fig.add_subplot(ax3)
        ax3.scatter(zS,yS,color=UCgray[1], s = s0)
        ax3.scatter(zS[indexAV],yS[indexAV],color=colorBeh['AngleVelocity'], s = s1)
        ax3.scatter(zS[indexEW],yS[indexEW],color=colorBeh['Eigenworm3'], s = s2)
        ax3.set_ylabel('Y')
        ax3.set_xlabel('Z')
        

###############################################    
# 
# scatter plot of LASSO and other linear models
#
##############################################  
def scatterSingleLinearFit(fig, gridloc, pars, results, data, splits, behaviors):
    inner_grid = gridspec.GridSpecFromSubplotSpec(len(behaviors), 1,
                subplot_spec=gridloc, hspace=1, wspace=0.25)
    for lindex, label in enumerate(behaviors):
        #weights, intercept, alpha, _,_ = resultSet[key][fitmethod][label]
        weights = results[label]['weights']
        intercept = results[label]['intercepts']
        if pars['useRank']:
            x = data['Neurons']['rankActivity']
        else:
            x = data['Neurons']['Activity']
        y = data['Behavior'][label]
        trainingsInd, testInd = splits[label]['Train'], splits[label]['Test']
        # calculate y from model
        yPred = np.dot(weights, x) + intercept
        
        yTrain = np.ones(yPred.shape)*np.nan
        yTrain[trainingsInd] = yPred[trainingsInd]
        
        yTest =  np.ones(yPred.shape)*np.nan
        yTest[testInd] = yPred[testInd]
        
        #if random=='random':
        #    yTest = yPred
        # plot training and test set behavior and prediction
        ax1 = plt.Subplot(fig, inner_grid[lindex, 0])
        ax1.scatter(y[testInd], yTest[testInd])
        
        fig.add_subplot(ax1)        

def plotLinearModelScatter(dataSets, resultSet, keyList, pars, fitmethod='LASSO', behaviors = ['AngleVelocity', 'Eigenworm3'], random = 'none'):
    """make an overview figure with Lasso weights and components."""
    nWorms = len(keyList)
    
    fig = plt.figure(fitmethod,(2*6.8, nWorms*1.7*len(behaviors)))
    outer_grid = gridspec.GridSpec(nWorms, 1, hspace=0.25, wspace=0.25)
    
    for kindex, key in enumerate(keyList):
        gridloc = outer_grid[kindex, 0]
        
        data = dataSets[key]
        
        results = resultSet[key][fitmethod]
        splits = resultSet[key]['Training']
        scatterSingleLinearFit(fig, gridloc, pars, results, data, splits, behaviors)
        
def averageResultsLinear(resultSets1,resultSets2, keyList1, keyList2, fitmethod = "LASSO",  behaviors = ['AngleVelocity', 'Eigenworm3']):
    """show box plots and paired plots for results."""
    fig = plt.figure('Results {}'.format(fitmethod),(6.8,6.8))
    gs = gridspec.GridSpec(2, 3, hspace=0.25, wspace=0.25, width_ratios=[1,1,2])
    keyListsAll = [keyList1, keyList2]
    resultSetsAll= [resultSets1,resultSets2]
    for i in range(2):
        keyList= keyListsAll[i]
        resultSets = resultSetsAll[i]
        # plot paired R2 results for multiple neurons
        for lindex, label in enumerate(behaviors):
            print [np.concatenate([resultSets[key][fitmethod][label]['individualScore']]) for key in keyList]
            
            r2s = [[np.max(np.concatenate([resultSets[key][fitmethod][label]['individualScore']])), resultSets[key][fitmethod][label]['cumulativeScore'][-1]] for key in keyList if len(resultSets[key][fitmethod][label]['individualScore'])>0]
            r2s =np.array(r2s)
            
            ax1 = plt.subplot(gs[i, lindex])
            ax1.set_ylabel("{} $R^2$".format(fitmethod))
            ax1.plot(np.ones(len(r2s))*0, r2s[:,0], 'o', color=colorBeh[label])
            ax1.plot(np.ones(len(r2s))*1, r2s[:,1], 'o', color=colorBeh[label])
            ax1.plot(r2s.T, 'k-')
            ax1.set_xticks([0,1])
            #ax1.set_ylim([0,1])
            ax1.set_xlim([-0.5,1.5])
            ax1.set_xticklabels(['best single neuron', 'group of neurons'],  rotation=30)
            
        Ns = np.array([np.array([resultSets[key][fitmethod][label]['noNeurons'] for label in behaviors]) for key in keyList])
        print Ns.shape
        colors = np.array([colorBeh[label] for label in behaviors])
        labels = np.array([names[label] for label in behaviors])
        
        ax2 = plt.subplot(gs[i, 2])
        ax2.set_ylabel("Number of neurons")
        mkStyledBoxplot(fig, ax2, [0,1], Ns.T, colors, labels)
        ax2.set_xlim([-0.5,1.5])
    gs.tight_layout(fig)
    
    
def mkStyledBoxplot(ax, x_data, y_data, clrs, lbls) : 
    
    dx = np.min(np.diff(x_data))
    
    for xd, yd, cl in zip(x_data, y_data, clrs) :
       
        bp = ax.boxplot(yd, positions=[xd], widths = 0.2*dx, \
                        notch=False, patch_artist=True)
        plt.setp(bp['boxes'], edgecolor=cl, facecolor=cl, \
             linewidth=1, alpha=0.4)
        plt.setp(bp['whiskers'], color=cl, linestyle='-', linewidth=1, alpha=1.0)    
        for cap in bp['caps']:
            cap.set(color=cl, linewidth=1)       
        for flier in bp['fliers']:
            flier.set(marker='+', color=cl, alpha=1.0)            
        for median in bp['medians']:
            median.set(color=cl, linewidth=1) 
        jitter = (np.random.random(len(yd)) - 0.5)*dx / 20 
        dotxd = [xd - 0.25*dx]*len(yd) + jitter

        # make alpha stronger
        ax.plot(dotxd, yd, linestyle='None', marker='o', color=cl, \
                markersize=3, alpha=0.5)  
#    ymin = min([min(m) for m in y_data])
#    ymax = max([max(m) for m in y_data])
#    dy = 10 ** np.floor(np.log10(ymin))
#    ymin, ymax = ymin-dy, ymax+dy
#    xmin, xmax = min(x_data)-0.5*dx, max(x_data)+0.5*dx
#    ax.set_xlim(xmin, xmax)        
#    ax.set_ylim(ymin, ymax)  
    ax.set_xticks(x_data)

#    for loc, spine in ax.spines.items() :
#        if loc == 'left' :
#            spine.set_position(('outward', 0))  # outward by 5 points
#            spine.set_smart_bounds(True)
#        elif loc == 'bottom' :
#            spine.set_position(('outward', 5))  # outward by 5 points
#            spine.set_smart_bounds(True)            
#        else :
#            spine.set_color('none')  # don't draw spine
    ax.yaxis.set_ticks_position('left') # turn off right ticks
    ax.xaxis.set_ticks_position('bottom') # turn off top ticks
    ax.get_xaxis().set_tick_params(direction='out')
    ax.patch.set_facecolor('white') # ('none')
    ax.set_xticklabels(lbls, rotation=30)
    
    #ax.set_aspect(2.0 / (0.1*len(lbls)), adjustable=None, anchor=None)
    #ax.set_aspect(0.01 / (len(y_data)), adjustable=None, anchor=None)        

def averageResultsPCA( resultSetsAll, keyListsAll,labels,colors,fitmethod = "PCA"):
    """show box plots and paired plots for results."""
    fig = plt.figure('Results {}'.format(fitmethod),(6.8,3.4))
    gs = gridspec.GridSpec(1, 2, hspace=0.25, wspace=0.25)
    #keyListsAll = [keyList1, keyList2]
    #resultSetsAll= [resultSets1,resultSets2]
    
    # plot paired R2 results for multiple neurons
    r2s = []
    for ki,keyList in enumerate(keyListsAll):
        resultSets = resultSetsAll[ki]
        r2s.append([resultSets[key][fitmethod]['expVariance'][0] for key in keyList])
    
    cumr2s = []
    for ki,keyList in enumerate(keyListsAll):
        resultSets = resultSetsAll[ki]
        cumr2s.append([np.cumsum(resultSets[key][fitmethod]['expVariance'])[-1] for key in keyList])
    
    ydata= [r2s, cumr2s]
    titles = ['first component', 'first {} components'.format(resultSets[key][fitmethod]['nComp'])]
    for i in range(2):
        
        y =np.array(ydata[i])
        
        ax1 = plt.subplot(gs[i])
        ax1.set_title(titles[i])
        ax1.set_ylabel("{} explained variance".format(fitmethod))
        
        mkStyledBoxplot(fig, ax1, range(len(labels)), y, colors, labels)
        #ax1.plot(np.ones(len(r2s))*0, r2s[:,0], 'o', color=colors[i], label=labels[i])
        #ax1.plot(np.ones(len(r2s))*1, r2s[:,1], 'o', color=colors[i])
        #ax1.plot(r2s.T, 'k-')
        ax1.legend()
        #ax1.set_xticks([0,1])
        ax1.set_xlim([-0.5,len(labels)+0.5])
        #ax1.set_xticklabels(['best component', 'first {} components'.format(resultSets[key][fitmethod]['nComp'])],  rotation=30)
        ax1.set_ylim([0,1])
            
        
        
    gs.tight_layout(fig)
    
def plotNeuronPredictedFromBehavior(results, data):
    label = 'AngleVelocity'
    splits = results['Training']
    train, test = splits[label]['Train'], splits[label]['Test']
    res = results['RevPred']
    newHM = res['predictedNeuralDynamics']
    orderedWeights = res['behaviorWeights'][:,res['behaviorOrder']]
    # plot stuff
    plt.figure('PredictedNeuralActivity', figsize=(2.28*4,2.28*6))
    plt.subplot(321)
    #show original heatmap
    plotHeatmap(data['Neurons']['Time'], data['Neurons']['Activity'])
    # show reduced dimensionality heatmap
    plotHeatmap(data['Neurons']['Time'][test], res['lowDimNeuro'][:,test])
    plt.subplot(322)
    plotHeatmap(data['Neurons']['Time'][test], newHM[:,test], vmin=np.min(newHM)*1.1, vmax=np.max(newHM)*0.9)
    plt.subplot(324)
    for ind, i in enumerate(res['PCA_indices'][:4]):
        x = data['Neurons']['Time'][test]
        line1, = plt.plot(x, res['NeuralPCS'][test,i]+ind*12, color='C0', label='Neural PCs')
        line2, = plt.plot(x, res['predictedNeuralPCS'][test,i]+ind*12, color='C3', label= 'Predicted')
        plt.text(x[-1]*0.9, 1.2*np.max(res['predictedNeuralPCS'][test,i]+ind*10), '$R^2={:.2f}$'.format(res['R2_test'][ind]))
    plt.legend([line1, line2], ['Neural PCs', 'Predicted from Behavior'], loc=2)
    ylabels = ['PC {}'.format(index+1) for index in res['PCA_indices'][:4]]
    plt.yticks(np.arange(0,4*12, 12), ylabels)
    plt.xlabel('Time(s)')
    plt.subplot(323)
    for ind, i in enumerate(res['behaviorOrder']):
        plt.plot(data['Neurons']['Time'], res['behavior'][:,i]+ind*4, color='k', label = res['behaviorLabels'][i], alpha=0.35+0.1*ind)
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
    plt.tight_layout()
    plt.show()
    # order the map by the components that is most explained by behavior
    #ci = np.argmax(explained_variance_score(pcs[test],predN[test],multioutput ='raw_values'))
    #indices = np.argsort(explained_variance_score(pcs[test],predN[test],multioutput ='raw_values'))
    #print len(indices), predN.shape
    #print lin.coef_.shape
    # calculate how good the heatmap is -- explained variance of the fit first
    #print explained_variance_score(pcs[test],predN[test],multioutput ='raw_values')
    #print explained_variance_score(pcs[train],predN[train],multioutput ='raw_values')
    #print explained_variance_score(data['Neurons']['Activity'], newHM)
    #print explained_variance_score(pca.inverse_transform(pcs).T[:,test], newHM[:,test])
    #print explained_variance_score(pca.inverse_transform(pcs).T, newHM)