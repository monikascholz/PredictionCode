# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 15:47:34 2018

@author: monika
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:15:14 2018
Figure 3 - Sparse linear model predicts behavior
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
from  sklearn.metrics.pairwise import pairwise_distances
from matplotlib_venn import venn2
 
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
# custom pip
#import svgutils as svg
#
import makePlots as mp
import dataHandler as dh
import dimReduction as dr

from stylesheet import *
################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML32', 'AML70']:
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
#  use registration to find candidate neurons
#
################################################
ventral = [1,1,1,1,1,-1]
movingAML32 = 'BrainScanner20170613_134800'
moving = data['AML32_moving']['input'][movingAML32]
flag = 'ElasticNet'

from pycpd import deformable_registration, rigid_registration, affine_registration
index = 0
markers = ['p', '^', '*', 'X', '+', '8', 's']
# use the moving dataset as reference
Xref = moving['Neurons']['Positions'].T
Xref -=np.mean(Xref,axis=0)
# load atlas data
neuron2D = 'utility/celegans277positionsKaiser.csv'
labels = np.array(np.loadtxt(neuron2D, delimiter=',', usecols=(0), dtype=str))
neuronAtlas2D = np.loadtxt(neuron2D, delimiter=',', usecols=(1,2))
relevantIds = (neuronAtlas2D[:,0]>-0.1)#*(Xref[:,0]<0.1)
A = neuronAtlas2D[relevantIds]
A[:,0] = -A[:,0]
labels = labels[relevantIds]
A -=np.mean(A, axis=0)
A /= np.ptp(A, axis=0)
A*= 1.2*np.ptp(Xref[:,:2], axis=0)
# register atlas to reference dataset
registration = rigid_registration
reg = registration(Xref[:,:2], A, tolerance=1e-5)
reg.register(callback=None)
registration =deformable_registration

reg = registration(Xref[:,:2],reg.TY, tolerance=1e-5)
def callback(iteration, error, X, Y):
    return 0
reg.register(callback)
A = reg.TY


Candidates = {}
# save some interesting neurons
for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
        dset = data[key]['input']
        res = data[key]['analysis']
        for idn in dset.keys():
            print idn
            X = np.copy(dset[idn]['Neurons']['Positions']).T
            X -=np.mean(Xref,axis=0)
            X[:,1] *=ventral[index]
            xS, yS, _ = X.T
            registration = rigid_registration
            
            reg = registration(Xref, X, tolerance=1e-5)
            reg.register(callback=None)
            registration = deformable_registration
            
            reg = registration(Xref, reg.TY, tolerance=1e-5)
            reg.register(callback=None)
            xS, yS, zS = reg.TY.T
            
            avWeights = res[idn][flag]['AngleVelocity']['weights']
            avRelevant = np.where(np.abs(avWeights>0))[0]
            avRelevant = avRelevant[np.argsort(np.abs(avWeights[avRelevant]))]
            # check if prediction is good otehrwise pass
            if res[idn][flag]['AngleVelocity']['scorepredicted']<0.2:
                avRelevant = []
            tWeights = res[idn][flag]['Eigenworm3']['weights']
            tRelevant = np.where(np.abs(tWeights>0))[0]
            tRelevant = tRelevant[np.argsort(np.abs(tWeights[tRelevant]))]
            # check if prediction is good otherwise pass
            if res[idn][flag]['Eigenworm3']['scorepredicted']<0.2:
                tRelevant = []
            
            n = -1
            radius = 3 
            for relevant in [avRelevant[-n:], tRelevant[-n:]]:
                if len(relevant) >0:
                    D = pairwise_distances(np.vstack([xS[relevant], yS[relevant]]).T, A)
                    print D.shape
                    # find minimal distances - this is the atlas ID
                    candNeurons = np.where(D<radius)[1]
                    
                    print labels[candNeurons]
            
         
           
            
            
            
                   