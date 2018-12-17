# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 12:11:31 2018
load all matlab datasets in a hdf5 format with standard preprocessing.
@author: monika
"""
import dataHandler as dh
#%%
###############################################    
# 
#    load data into dictionary
#
##############################################  
typ='AML70'


# GCamp6s; lite-1
if typ =='AML70': 
    folder = "AML70_moving/{}_MS/"
    dataLog = "AML70_moving/AML70_datasets.txt"
    
# GCamp6s 
if typ =='AML32': 
    folder = "AML32_moving/{}_MS/"
    dataLog = "AML32_moving/AML32_datasets.txt"

##### GFP
elif typ =='AML18': 
    folder = "AML18_moving/{}_MS/"
    dataLog = "AML18_moving/AML18_datasets.txt"

# GCamp -- immobilized
elif typ =='AML32imm': 
    folder = "AML32_immobilized/{}_MS/"
    dataLog = "AML32_immobilized/AML32_immobilized_datasets.txt"

# data parameters
dataPars = {'medianWindow':3, # smooth eigenworms with gauss filter of that size, must be odd
            'savGolayWindow':5, # savitzky-golay window for angle velocity derivative. must be odd
            'rotate':True, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5 # gauss window for red and green channel
            }

outLoc = 'Data_{}.hdf5'.format(typ)
dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars)

#%%
###############################################    
# 
# save data as HDF5 file
#
##############################################
dh.saveDictToHDF(outLoc, dataSets)