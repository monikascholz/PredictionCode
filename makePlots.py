# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:50:38 2017
plot assistant. make pretty plots.
@author: monika
"""
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score, r2_score
axescolor = 'k'
mpl.rcParams["axes.edgecolor"]=axescolor
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

mpl.rcParams["text.color"]='k'
mpl.rcParams["ytick.color"]=axescolor
mpl.rcParams["xtick.color"]=axescolor
mpl.rcParams["axes.labelcolor"]='k'
mpl.rcParams["savefig.format"] ='pdf'
#mpl.rcParams['text.usetex'] =True
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

# continous behavior colors
colorBeh = {'AngleVelocity':'#DC143C', 'Eigenworm3':'#4876FF'}
# discrete behaviors
colDict = {-1:'red',0:'k',1:'green',2:'blue'}
labelDict = {-1:'Reverse',0:'Pause',1:'Forward',2:'Turn'}

ethocmap = mpl.colors.ListedColormap([mpl.colors.to_rgb('#C21807'), UCgray[1], mpl.colors.to_rgb('#4AA02C'), mpl.colors.to_rgb('#0F52BA')], name='etho', N=None)

            
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

def multicolor(ax,x,y,z,t,c, threedim = True):
    """multicolor plot from francesco."""
    if threedim:
        points = np.array([x,y,z]).transpose().reshape(-1,1,3)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = Line3DCollection(segs, cmap=c, lw=0.5)
        lc.set_array(t)
        ax.add_collection3d(lc)
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_ylim(np.min(y),np.max(y))
        ax.set_zlim(np.min(z),np.max(z))
    else:
        points = np.array([x,y]).transpose().reshape(-1,1,2)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = LineCollection(segs, cmap=c, lw=0.5)
        lc.set_array(t)
        ax.add_collection(lc)
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_ylim(np.min(y),np.max(y))

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
    plt.plot(T, E, color = color, lw=0.5)
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
    nWorms = len(dataSets)
    fig = plt.figure('Overview',(6.8, nWorms*3.4))
    gs = gridspec.GridSpec(nWorms*2, 3,
                           width_ratios=[1,0.1, 2])
    for dindex, key in enumerate(keyList):
        print 'Plotting overview of ', key
        data = dataSets[key]
        ax = plt.subplot(gs[2*dindex:2*dindex+2,0])
        hm = plotHeatmap(data['Neurons']['Time'], data['Neurons']['rankActivity'])
        ax2 = plt.subplot(gs[2*dindex:2*dindex+2,1])
        plt.colorbar(hm, cax=ax2)
        ax3 = plt.subplot(gs[2*dindex,2])
        yMin, yMax = np.min(data['Behavior']['AngleVelocity']), np.max(data['Behavior']['AngleVelocity'])
        print len(data['Neurons']['Time']),  len(data['Behavior']['Ethogram'])
        plotEthogram(ax3, data['Neurons']['Time'],  data['Behavior']['Ethogram'], alpha = 0.25, yValMin=yMin,yValMax=yMax )        
        plotEigenworms(data['Neurons']['Time'], data['Behavior']['AngleVelocity'], color = colorBeh['AngleVelocity'],label = 'Angular velocity')
        
        ax4 = plt.subplot(gs[2*dindex+1,2])
        yMin, yMax = np.nanmin(data['Behavior']['Eigenworm3']), np.nanmax(data['Behavior']['Eigenworm3'])
        plotEthogram(ax4, data['Neurons']['Time'],  data['Behavior']['Ethogram'], alpha = 0.25,yValMin=yMin,yValMax=yMax)
        plotEigenworms(data['Neurons']['Time'], data['Behavior']['Eigenworm3'],color = colorBeh['Eigenworm3'], label = 'Turns')
    plt.tight_layout()
    plt.show()
    


def plotNeurons3D(dataSets, keyList, threed = True):
    """plot neuron locations."""
    nWorms = len(dataSets)
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
# neuronal signal pca, first n components and weights plotted
#
############################################## 
def plotPCAresults(dataSets, resultSet, keyList):
    """make an overview figure with PCA weights and components."""
    nWorms = len(keyList)
    fig = plt.figure('PCA',(6.8, nWorms*3.4))
    outer_grid = gridspec.GridSpec(nWorms, 1, hspace=0.25, wspace=0.25)
    
    for kindex, key in enumerate(keyList):
        data = dataSets[key]
        results = resultSet[key]['PCA']
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 3,
            subplot_spec=outer_grid[kindex], hspace=0.5, wspace=0.35, width_ratios=[2,0.2, 1])
        
        ax1 = plt.Subplot(fig, inner_grid[0, 0])
        # plot neurons ordered by weight in first PCA component
        cax1 = plotHeatmap(data['Neurons']['Time'],  data['Neurons']['Activity'][results['neuronOrderPCA']], ax= ax1)
        ax1.set_xlabel('Time (s)')
        fig.add_subplot(ax1)        
        axcb = plt.Subplot(fig, inner_grid[0, 1])        
        plt.colorbar(cax1, cax=axcb, use_gridspec = True)
        fig.add_subplot(axcb) 
        
        # plot the weights
        ax2 = plt.Subplot(fig, inner_grid[0,2])
        pcs = results['neuronWeights']
        rank = np.arange(0, len(pcs))
        print len(pcs[:,0])
        ax2.fill_betweenx(rank, np.zeros(len(data['Neurons']['Activity'])),pcs[:,0][results['neuronOrderPCA']], step='pre')
        ax2.fill_betweenx(rank, np.zeros(len(data['Neurons']['Activity'])),pcs[:,1][results['neuronOrderPCA']], step='pre')       
        ax2.fill_betweenx(rank, np.zeros(len(data['Neurons']['Activity'])),pcs[:,2][results['neuronOrderPCA']], step='pre')       
        ax2.set_xlabel('Neuron weight')
        ax2.spines['left'].set_visible(False)
        ax2.set_yticks([])
        fig.add_subplot(ax2)
        
        ax3 = plt.Subplot(fig, inner_grid[1,0])
        for i in range(3):
            ax3.plot(data['Neurons']['Time'], 0.1*i+results['pcaComponents'][i], label=i, lw=0.5)
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
        
    outer_grid.tight_layout(fig)
    
###############################################    
# 
# neuronal signal pca, plot in 3D with behavior labels
#
############################################## 
def plotPCAresults3D(dataSets, resultSet, keyList, col = 'phase'):
    """Show neural manifold with behavior label."""
    nWorms = len(keyList)
    fig1 = plt.figure('PCA manifold',(6.8, nWorms*3.4))
    outer_grid = gridspec.GridSpec(nWorms, 1, hspace=0.25, wspace=0.25)
    fig2 = plt.figure('PCA projections',(6.8, nWorms*3.4))
    for kindex, key in enumerate(keyList):
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 3,
            subplot_spec=outer_grid[kindex], hspace=0.5, wspace=0.35)
        data = dataSets[key]
        results = resultSet[key]['PCA']
        x,y,z = results['pcaComponents'][:3,]
        
        if col == 'phase':
            colorBy = np.arctan2(data['Behavior']['Eigenworm2'],data['Behavior']['Eigenworm1'])/np.pi
            cm = cyclon
        elif col == 'velocity':
            colorBy = data['Behavior']['AngleVelocity']
            cm = 'jet'
        elif col =='turns':
            colorBy = data['Behavior']['Eigenworm3']
            cm = 'jet'
        else:
            colorBy = np.reshape(np.array(data['Behavior']['Ethogram']), (-1, ))
            cm = ethocmap
        #etho[np.isnan(etho)] = 1
        #print ethocmap[0], ethocmap[1]
        #ax3 = plt.Subplot(fig, outer_grid[kindex])
        ax4= fig1.add_subplot(outer_grid[kindex], projection = '3d')
        multicolor(ax4,x,y,z,colorBy,c = cm, threedim = True)
        ax4.set_ylabel('PCA components 2')
        ax4.set_xlabel('PCA components 1')
        ax4.set_zlabel('PCA components 3')
        
        ax1 = plt.Subplot(fig2, inner_grid[0])
        fig2.add_subplot(ax1)
        multicolor(ax1,x,y,z,colorBy,c = cm,threedim = False)
        ax1.set_ylabel('Y')
        ax1.set_xlabel('X')
        
        ax2 = plt.Subplot(fig2, inner_grid[1])
        fig2.add_subplot(ax2)
        multicolor(ax2,x,z,z,colorBy,c = cm,threedim = False)
        ax2.set_ylabel('Z')
        ax2.set_xlabel('X')
        
        ax3 = plt.Subplot(fig2, inner_grid[2])
        fig2.add_subplot(ax3)
        multicolor(ax3,z,y,x,colorBy,c = cm,threedim = False)
        ax3.set_ylabel('Y')
        ax3.set_xlabel('Z')
        
        
        
    #outer_grid.tight_layout(fig2)

   
    
###############################################    
# 
# linear prediction from single neurons
#
##############################################    
def plotLinearPredictionSingleNeurons(dataSets, resultSet, keyList):
    """make an overview figure with linear regression predictions."""
    nWorms = len(keyList)
    
    fig = plt.figure('Linear Prediction',(6.8, nWorms*3.4))
    outer_grid = gridspec.GridSpec(nWorms, 1, hspace=0.25, wspace=0.25)
    
    for kindex, key in enumerate(keyList):
        trainingsInd, testInd = resultSet[key]['Training']['Indices']  
        data = dataSets[key]
        inner_grid = gridspec.GridSpecFromSubplotSpec(4, 2,
                subplot_spec=outer_grid[kindex], hspace=0.5, wspace=0.35, width_ratios=[1, 3], height_ratios = [1,3,1,3])
        for lindex, label in enumerate(['AngleVelocity', 'Eigenworm3']):
            results = resultSet[key]['Linear Regression'][label]
            # plot r2 predictive scores for each neuron
            ax1 = plt.Subplot(fig, inner_grid[2*lindex:2*lindex+2,0])
            scores = results['score']
            scorepred = results['scorepredicted']
            
            # order neurons by r^2 score
            indices = np.arange(len(data['Neurons']['Activity']))
            indices = np.argsort(scores)
            
            #ax1.fill_between(np.arange(len(scores)),np.zeros(len(scores)),scores[indices], step='pre')
            #ax1.plot(np.arange(len(scores)),scores[indices])
            #ax1.plot(scores[indices])
            hist, bins = np.histogram(scores, bins = 30, density = True)
            ax1.fill_between(bins[:-1],np.zeros(len(hist)), hist, step='post', color=colorBeh[label])
            hist, bins = np.histogram(scorepred, bins = 30, density = True)
            ax1.fill_between(bins[:-1],np.zeros(len(hist)), hist, step='post',color=colorBeh[label], alpha = 0.5)
            ax1.set_xlabel(r'$R^2$ score')
            ax1.set_ylabel('PDF($R^2$ score)')
            #ax1.set_ylim([-1,1])
            #ax1.set_xlim([0,100])
            fig.add_subplot(ax1)
            
#            #plot most predictive neurons
#            ax2 = plt.Subplot(fig, inner_grid[2*lindex,1])
#            for i in range(3):
#                ax2.plot(data['Neurons']['Time'][testInd], data['Neurons']['Activity'][indices[-i-1],testInd], label=i)
#            ax2.set_ylabel('Neural Activity')
#            ax2.set_xlabel('Time (s)')
#            fig.add_subplot(ax2)
            # plot ground truth and best prediction 
            #---- only predicted part
#            ax3 = plt.Subplot(fig, inner_grid[2*lindex:2*lindex+2,1])
#            y = data['Behavior'][label][testInd]
#            x = data['Neurons']['Activity'][:,testInd]
#            ax3.plot(data['Neurons']['Time'][testInd], y, color='red', label = 'Behavior')
#            for k, ind in enumerate(indices[-3:]):
#                
#                slope, intercept = results[0,ind],results[1,ind]
#                ax3.plot(data['Neurons']['Time'][testInd], x[ind]*slope + intercept, color='k', alpha = 0.2*k+0.2, label = "predictive score: {:.2f} fitted score: {:.2f}".format(scorepred[ind], scores[ind]))
#                ax3.set_xlim(np.percentile(data['Neurons']['Time'][testInd], [0,100]))
#            ax3.legend()
#            fig.add_subplot(ax3)
            #---- whole trace
            ax3 = plt.Subplot(fig, inner_grid[2*lindex:2*lindex+2,1])
            y = data['Behavior'][label]
            x = data['Neurons']['Activity']
            ax3.plot(data['Neurons']['Time'], y, color=colorBeh[label], label = 'Behavior')
            for k, ind in enumerate(indices[-3:]):
                # best three neurons
                slope, intercept = results['score'][ind],results['scorepredicted'][ind]
                ax3.plot(data['Neurons']['Time'], x[ind]*slope + intercept, color='k', alpha = 0.2*k+0.2, label = "predictive score: {:.2f} fitted score: {:.2f}".format(scorepred[ind], scores[ind]))
                ax3.set_xlim(np.percentile(data['Neurons']['Time'], [0,100]))
            ax3.legend()
            fig.add_subplot(ax3)
        
    outer_grid.tight_layout(fig)
    
###############################################    
# 
# plot LASSO and other linear models
#
##############################################    
def plotLinearModelResults(dataSets, resultSet, keyList, fitmethod='LASSO'):
    """make an overview figure with Lasso weights and components."""
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
            alpha = resultSet[key][fitmethod][label]['alpha']
            
            y = data['Behavior'][label]
            x = data['Neurons']['Activity']
            # calculate y from model
            yPred = np.dot(weights, x) + intercept
            # calculate relevant neurons
            Xvelo = np.copy(x)
            Xvelo[weights==0,:] = 0 
            
            # plot training and test set behavior and prediction
            ax1 = plt.Subplot(fig, inner_grid[lindex, 0])
            ax1.plot(data['Neurons']['Time'], y, color=colorBeh[label], label = 'Behavior', lw=0.5, alpha = 0.5)
            ax1.plot(data['Neurons']['Time'][trainingsInd], yPred[trainingsInd], color=colorBeh[label], label = 'Training')
            ax1.plot(data['Neurons']['Time'][testInd], yPred[testInd], color='k', label = 'Prediction')
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
# plot how performance increases when adding more neurons
#
##############################################    
def plotLinearModelProgression(dataSets, resultSet, keyList, fitmethod='LASSO'):
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
    nWorms = len(dataSets)
    fig = plt.figure('Neurons',(6.8, nWorms*3.4))
    gs = gridspec.GridSpec(nWorms,1, hspace=0.25, wspace=0.25)
    for dindex, key in enumerate(keyList):
        data = dataSets[key]
        xS, yS, zS = data['Neurons']['Positions']
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