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
#folder = "SelectDatasets/BrainScanner20170610_105634_linkcopy/"
#folder = "AML32_moving/{}_MS/"
#dataLog = "AML32_moving/AML32_datasets.txt"
#
## output is stored here
#outfile = "SelectDatasets/test.npz"
#
#dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder)
#  
#keyList = np.sort(dataSets.keys())
#
# try to load CLs
# load centerline data eigenproj
#folder = folder.format(keyList[0])
#cl = dh.loadCenterlines(folder)
###############################################    
# 
#    try out hdf5
#
############################################## 
print h5py.__version__
f = h5py.File("mytfile3.hdf5", mode='w')
f.create_group("test")
t = np.arange(10)
f['test'].create_dataset("ArrayName",data = t)

f.close()