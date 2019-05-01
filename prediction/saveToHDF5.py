# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:31:32 2018
save data as hdf5 file.
@author: monika
"""

# standard modules
import numpy as np
import matplotlib.pylab as plt
import h5py
# custom modules 
import dataHandler as dh
import makePlots as mp
import dimReduction as dr

###############################################    
# 
#    load data into dictionary
#
##############################################  

typ='AML70imm'
# GCamp6s; lite-1
if typ =='AML70': 
    folder = "AML70_moving/{}_MS/"
    dataLog = "AML70_moving/AML70_datasets.txt"
    outLoc = "AML70_moving/Analysis/"
# GCamp6s 
if typ =='AML32': 
    folder = "AML32_moving/{}_MS/"
    dataLog = "AML32_moving/AML32_datasets.txt"
    outLoc = "AML32_moving/Analysis/"
##### GFP
elif typ =='AML18': 
    folder = "AML18_moving/{}_MS/"
    dataLog = "AML18_moving/AML18_datasets.txt"
    outLoc = "AML18_moving/Analysis/"
# immobile GCamp6
elif typ =='AML32imm': 
    folder = "AML32_immobilized/{}_MS/"
    dataLog = "AML32_immobilized/AML32_immobilized_datasets.txt"
    
# immobile GCamp6 -lite-1
elif typ =='AML70imm': 
    folder = "AML70_immobilized/{}_MS/"
    dataLog = "AML70_immobilized/AML70imm_datasets.txt"
    
# data parameters
dataPars = {'medianWindow':5, # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow':10, # sgauss window for angle velocity derivative. must be odd
            'rotate':True, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5  # gauss window for red and green channel
            }


dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars, nDatasets = None)
keyListAll = np.sort(dataSets.keys())
print keyListAll