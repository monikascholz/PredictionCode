# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:50:38 2017
plot assistant. make pretty plots.
@author: monika
"""
import numpy as np
import matplotlib as mpl
#
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter1d
#
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
mpl.rcParams["axes.labelsize"]=  12
mpl.rcParams["xtick.labelsize"]=  12
mpl.rcParams["ytick.labelsize"]=  12
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
colorBeh = {'AngleVelocity':'#DC143C', 'Eigenworm3':'#4876FF', 'Eigenworm2':'#4caf50'}
# continous behavior colors - prediction
colorPred = {'AngleVelocity':'#6e0a1e', 'Eigenworm3':'#1c2f66', 'Eigenworm2':'#265728'}
# discrete behaviors
colDict = {-1:'red',0:'k',1:'green',2:'blue'}
labelDict = {-1:'Reverse',0:'Pause',1:'Forward',2:'Turn'}
# color the ethogram
ethocmap = mpl.colors.ListedColormap([mpl.colors.to_rgb('#C21807'), UCgray[1], mpl.colors.to_rgb('#4AA02C'), mpl.colors.to_rgb('#0F52BA')], name='etho', N=None)
ethobounds=[-1,0,1,2, 3]
ethonorm = mpl.colors.BoundaryNorm(ethobounds, ethocmap.N)

# rename behaviors for plots
names = {'AngleVelocity': 'Angular velocity',
         'Eigenworm3': 'Turns', 
         'Eigenworm2': 'Head swing'
        }
            
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
    x = x[::cg]
    y = y[::cg]
    z = z[::cg]
    t = t[::cg]
    if threedim:
        points = np.array([x,y,z]).transpose().reshape(-1,1,3)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = Line3DCollection(segs, cmap=c, lw=0.5)
        if etho:
            lc = Line3DCollection(segs, cmap=c, lw=0.5, norm=ethonorm)
        lc.set_array(t)
        ax.add_collection3d(lc)
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_ylim(np.min(y),np.max(y))
        ax.set_zlim(np.min(z),np.max(z))
    else:
        points = np.array([x,y]).transpose().reshape(-1,1,2)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = LineCollection(segs, cmap=c, lw=0.5)
        if etho:
            lc = LineCollection(segs, cmap=c, lw=0.5, norm=ethonorm)
        lc.set_array(t)
        ax.add_collection(lc)
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_ylim(np.min(y),np.max(y))
    return lc

def circle_scatter(axes, x_array, y_array, radius=0.5, **kwargs):
    """make scatter plot with axis unit radius.(behaves nice when zooming in)"""
    for x, y in zip(x_array, y_array):
        circle = plt.Circle((x,y), radius=radius, **kwargs)
        axes.add_patch(circle)
    return True

def plotHeatmap(T, Y, ax = None):
    """nice looking heatmap for neural dynamics."""
    if ax is None:
        ax = plt.gca()
    cax1 = ax.imshow(Y, aspect='auto', interpolation='none', origin='lower',extent=[0,T[-1],len(Y),0])#,vmax=2, vmin=-2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel("Neuron")
    return cax1
    
def plotEigenworms(T, E, label, color = 'k'):
    """make an eigenworm plot"""
    plt.plot(T, E, color = color, lw=1)
    plt.ylabel(label)
    plt.xlabel('Time (s)')

def plotEthogram(ax, T, etho, alpha = 0.5, yValMax=1, yValMin=0, legend=0):
    """make a block graph ethogram for elegans behavior"""
    colDict = {-1:'red',0:'k',1:'green',2:'blue'}
    labelDict = {-1:'Reverse',0:'Pause',1:'Forward',2:'Turn'}
    #y1 = np.where(etho==key,1,0)
    for key in colDict.keys():
        plt.fill_between(T, y1=np.ones(len(T))*yValMin, y2=np.ones(len(T))*yValMax, where=(etho==key)[:,0], \
        interpolate=False, color=colDict[key], label=labelDict[key], alpha = alpha)
    plt.xlim([min(T), max(T)])
    plt.ylim([yValMin, yValMax])
    plt.xlabel('Time (s)')
    plt.yticks([])
    if legend:
        plt.legend(ncol=2)
    
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
        hm = plotHeatmap(data['Neurons']['Time'], data['Neurons']['rankActivity'])
        ax2 = plt.subplot(gs[2*dindex:2*dindex+2,1])
        plt.colorbar(hm, cax=ax2)
             
        
        ax3 = plt.subplot(gs[2*dindex,2])
        yMin, yMax = np.min(data['Behavior']['AngleVelocity']), np.max(data['Behavior']['AngleVelocity'])
        #print len(data['Neurons']['Time']),  len(data['Behavior']['Ethogram'])
        plotEthogram(ax3, data['Neurons']['Time'],  data['Behavior']['Ethogram'], alpha = 0.25, yValMin=yMin,yValMax=yMax )        
        plotEigenworms(data['Neurons']['Time'], data['Behavior']['AngleVelocity'], color = colorBeh['AngleVelocity'],label = names['AngleVelocity'])
        
        ax4 = plt.subplot(gs[2*dindex+1,2])
        yMin, yMax = np.nanmin(data['Behavior']['Eigenworm3']), np.nanmax(data['Behavior']['Eigenworm3'])
        plotEthogram(ax4, data['Neurons']['Time'],  data['Behavior']['Ethogram'], alpha = 0.25,yValMin=yMin,yValMax=yMax)
        plotEigenworms(data['Neurons']['Time'], data['Behavior']['Eigenworm3'],color = colorBeh['Eigenworm3'], label = names['Eigenworm3'])
        #plot cms motion        
        ax5 = plt.subplot(gs[2*dindex:2*dindex+2,3])
        ax5.plot(data['Behavior']['X'], data['Behavior']['Y'])
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
def singlePCAResult(fig, gridloc, Neuro, results, time):
    """plot PCA of one dataset"""
    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 3,
        subplot_spec=gridloc, hspace=0.5, wspace=0.35, width_ratios=[2,0.2, 1])
    
    ax1 = plt.Subplot(fig, inner_grid[0, 0])
    # plot neurons ordered by weight in first PCA component
    cax1 = plotHeatmap(time,  Neuro[results['neuronOrderPCA']], ax= ax1)
    ax1.set_xlabel('Time (s)')
    fig.add_subplot(ax1)        
    axcb = plt.Subplot(fig, inner_grid[0, 1])        
    plt.colorbar(cax1, cax=axcb, use_gridspec = True)
    fig.add_subplot(axcb) 
    
    # plot the weights
    ax2 = plt.Subplot(fig, inner_grid[0,2])
    pcs = results['neuronWeights']
    rank = np.arange(0, len(pcs))
    
    ax2.fill_betweenx(rank, np.zeros(len(Neuro)),pcs[:,0][results['neuronOrderPCA']], step='pre')
    ax2.fill_betweenx(rank, np.zeros(len(Neuro)),pcs[:,1][results['neuronOrderPCA']], step='pre')       
    ax2.fill_betweenx(rank, np.zeros(len(Neuro)),pcs[:,2][results['neuronOrderPCA']], step='pre')       
    ax2.set_xlabel('Neuron weight')
    ax2.spines['left'].set_visible(False)
    ax2.set_yticks([])
    fig.add_subplot(ax2)
    
    ax3 = plt.Subplot(fig, inner_grid[1,0])
    for i in range(3):
        ax3.plot(time, 0.1*i+results['pcaComponents'][i], label=i, lw=0.5)
    ax3.set_ylabel('PCA components')
    ax3.set_xlabel('Time (s)')
    fig.add_subplot(ax3)
    ax4 = plt.Subplot(fig, inner_grid[1,2])
    nComp = results['nComp']
    
    ax4.fill_between(np.arange(nComp),results['expVariance'])
    ax4.step(np.arange(nComp),np.cumsum(results['expVariance']), where = 'pre')
    ax4.set_ylabel('Explained variance')
    ax4.set_xlabel('Number of components')
    fig.add_subplot(ax4)    
    
def plotPCAresults(dataSets, resultSet, keyList, pars, flag = 'PCA'):
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
        results = resultSet[key][flag]
        gridloc=outer_grid[kindex]
        singlePCAResult(fig, gridloc, Neuro, results, time)
        
    outer_grid.tight_layout(fig)        
    

    
###############################################    
# 
# neuronal signal pca, plot in 3D with behavior labels
#
############################################## 
def plotPCAresults3D(dataSets, resultSet, keyList,pars,  col = 'phase', flag = 'PCA', smooth = 6):
    """Show neural manifold with behavior label."""
    nWorms = len(keyList)
    print 'colored by ', col
    fig2 = plt.figure('{} projections'.format(flag),(6.8, nWorms*3.4))
    fig3 = plt.figure('{} temporal'.format(flag),(6.8, nWorms*3.4))
    outer_grid2 = gridspec.GridSpec(nWorms, 3, hspace=0.25, wspace=0.25)
    fig1 = plt.figure('{} manifold'.format(flag),(6.8, nWorms*3.4))
    outer_grid = gridspec.GridSpec(nWorms, 1, hspace=0.25, wspace=0.25)
    
    
    for kindex, key in enumerate(keyList):
        data = dataSets[key]
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2,
            subplot_spec=outer_grid[kindex], hspace=0.5, wspace=0.35, width_ratios=[1,0.2])
        
        inner_grid2 = gridspec.GridSpecFromSubplotSpec(3, 1,
            subplot_spec=outer_grid[kindex], hspace=0.5, wspace=0.35)
        
        results = resultSet[key][flag]
        x,y,z = results['pcaComponents'][:3,]
        x = gaussian_filter1d(x, smooth)
        y = gaussian_filter1d(y, smooth)
        z = gaussian_filter1d(z, smooth)
        
        etho=False
        if col == 'phase':
            colorBy = np.arctan2(data['Behavior']['Eigenworm2'],data['Behavior']['Eigenworm1'])/np.pi
            
            cm = cyclon
        elif col == 'velocity':
            colorBy = data['Behavior']['AngleVelocity']
            colorBy -= np.mean(data['Behavior']['AngleVelocity'])
            colorBy /= np.std(data['Behavior']['AngleVelocity'])
            cm = 'jet'
        elif col =='turns':
            colorBy = data['Behavior']['Eigenworm3']
            colorBy -= np.mean(data['Behavior']['Eigenworm3'])
            colorBy /= np.std(data['Behavior']['Eigenworm3'])
            cm = 'jet'
        elif col=='time':
            colorBy = data['Neurons']['Time']
            cm = 'jet'
        else:
            colorBy = np.reshape(np.array(data['Behavior']['Ethogram']), (-1, ))
            cm = ethocmap
            etho = True
        #etho[np.isnan(etho)] = 1
        #print ethocmap[0], ethocmap[1]
        #ax3 = plt.Subplot(fig, outer_grid[kindex])
        coarsegrain = 1
        ax4 = fig1.add_subplot(inner_grid[0,0], projection = '3d')
        cax = multicolor(ax4,x,y,z,colorBy,c = cm, threedim = True, etho=etho,  cg =coarsegrain)
        ax4.set_ylabel('{} components 2'.format(flag))
        ax4.set_xlabel('{} components 1'.format(flag))
        ax4.set_zlabel('{} components 3'.format(flag))
        if etho:
            
            #axcb = plt.Subplot(fig1, inner_grid[0, 1])     
            axcb = fig1.add_subplot(inner_grid[0, 1])
            cbar = plt.colorbar(cax, cax=axcb, use_gridspec = True, norm=ethonorm, drawedges=False)
            cbar.ax.get_yaxis().set_ticks([])
            for j, lab in enumerate(['Reverse','Pause','Forward','Turn']):
                cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center', color='white')
            cbar.ax.get_yaxis().labelpad = 15
            #fig1.add_subplot(axcb) 
        else:
            axcb = fig1.add_subplot(inner_grid[0, 1])
            cbar= plt.colorbar(cax, cax = axcb, use_gridspec=True)
        
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
        ax4 = fig3.add_subplot(inner_grid2[0])
        multicolor(ax4,data['Neurons']['Time'],x,x,colorBy,c = cm,threedim = False, etho=etho,  cg =coarsegrain)
        ax4.set_ylabel('PCA 1')
        ax4 = fig3.add_subplot(inner_grid2[1])
        multicolor(ax4,data['Neurons']['Time'],y,x,colorBy,c = cm,threedim = False, etho=etho,  cg =coarsegrain)
        
        ax4.set_ylabel('PCA 2')
        ax4 = fig3.add_subplot(inner_grid2[2])
        multicolor(ax4,data['Neurons']['Time'],z,x,colorBy,c = cm,threedim = False, etho=etho,  cg =coarsegrain)
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('PCA 3')
    outer_grid2.tight_layout(fig2)
    outer_grid.tight_layout(fig3)


def plotPCAcorrelates(dataSets, resultDict, keyList, pars, flag='PCA'):
    """correlate PCA with all sorts of stuffs"""
    nWorms = len(keyList)
    fig3 = plt.figure('{} Correlates'.format(flag),(6.8, nWorms*3.4))
    outer_grid = gridspec.GridSpec(nWorms, 1, hspace=0.25, wspace=0.25)

    for kindex, key in enumerate(keyList):
        data = dataSets[key]
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 5,
        subplot_spec=outer_grid[kindex], hspace=0.5, wspace=0.35)
        x,y,z = resultDict[key][flag]['pcaComponents'][:3,]
        
        theta = np.unwrap(np.arctan2(y, z))
        velo = dh.savitzky_golay(theta, window_size=17, order=5, deriv=1, rate=1)
        #velo = dh.savitzky_golay(x, window_size=7, order=3, deriv=2, rate=1)
        phase = np.arctan2(data['Behavior']['Eigenworm2'],data['Behavior']['Eigenworm1'])/np.pi
            
        size, a = 0.5, 0.1
        ax = fig3.add_subplot(inner_grid[0])
        ax.scatter(velo, data['Behavior']['CMSVelocity'], s=size, alpha=a)
        ax = fig3.add_subplot(inner_grid[1])
        ax.scatter(velo, data['Behavior']['AngleVelocity'], s=size, alpha=a)
        ax = fig3.add_subplot(inner_grid[2])
        ax.scatter(velo, data['Behavior']['Eigenworm1'], s=size, alpha=a)
        ax = fig3.add_subplot(inner_grid[3])
        ax.scatter(velo, phase, s=size, alpha=a)
        ax = fig3.add_subplot(inner_grid[4])
        ax.scatter(velo, phase, s=size, alpha=a)
    outer_grid.tight_layout(fig3)
        
###############################################    
# 
# plot LASSO and other linear models
#
##############################################  
def plotSingleLinearFit(fig, gridloc, pars, results, data, trainingsInd, testInd, behaviors):
    inner_grid = gridspec.GridSpecFromSubplotSpec(len(behaviors), 3,
                subplot_spec=gridloc, hspace=1, wspace=0.25, width_ratios=[3,1,1])
    for lindex, label in enumerate(behaviors):
        #weights, intercept, alpha, _,_ = resultSet[key][fitmethod][label]
        weights = results[label]['weights']
        intercept = results[label]['intercepts']
        if pars['useRank']:
            x = data['Neurons']['rankActivity']
        else:
            x = data['Neurons']['Activity']
        y = data['Behavior'][label]
       
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
        
        ax1.plot(data['Neurons']['Time'], yTrain, color=colorBeh[label], label = 'Training', alpha =0.4, lw=2)
        ax1.plot(data['Neurons']['Time'], y, color=colorBeh[label], label = 'Behavior', lw=1)
        ax1.plot(data['Neurons']['Time'], yTest, color=colorPred[label], label = r'$R^2$ {0:.2f}'.format(results[label]['scorepredicted']), lw=1)
        ax1.set_xlim(np.percentile(data['Neurons']['Time'], [0,100]))    
        ax1.set_ylabel(names[label])
        if lindex==len(behaviors)-1:
            ax1.set_xlabel('Time (s)')
        
        ax1.legend(loc=(0.0,0.9), ncol = 2)
        fig.add_subplot(ax1)
        
        # show how predictive each additional neuron is
        ax4 = plt.Subplot(fig, inner_grid[lindex, 2])
        ax4.plot(results[label]['cumulativeScore'], color=colorPred[label],marker='o',  markerfacecolor="none",markersize=5)
        ax4.plot(results[label]['individualScore'], color=colorBeh[label],marker='o', markerfacecolor="none", markersize=5)
          
        ax4.set_ylabel(r'$R^2$ score')
        if lindex==len(behaviors)-1:
            ax4.set_xlabel('Number of neurons')
        fig.add_subplot(ax4)
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
        trainingsInd, testInd = resultSet[key]['Training']['Indices']  
        data = dataSets[key]
        
        results = resultSet[key][fitmethod]
        plotSingleLinearFit(fig, gridloc, pars, results, data, trainingsInd, testInd, behaviors)
    #outer_grid.tight_layout(fig)
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
        s0,s1,s2 = 64, 32, 16 # size of gray, red, blue neurons
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