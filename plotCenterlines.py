# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 09:58:31 2017
plot centerlines.
@author: monika
"""
import matplotlib.pylab as plt
import dataHandler as dh
import numpy as np
folder = "AML32_moving/{}_MS/"
dataLog = "AML32_moving/AML32_datasets.txt"
dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder)
#  
# data parameters
dataPars = {'medianWindow':5, # smooth eigenworms with gauss filter of that size, must be odd
            'savGolayWindow':5, # savitzky-golay window for angle velocity derivative. must be odd
            'rotate':True, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5 # gauss window for red and green channel
            }

keyList = np.sort(dataSets.keys())
# load centerline data eigenproj
folder = folder.format('BrainScanner20170610_105634')
points = dh.loadPoints(folder)
#print points
cl = dh.loadCenterlines(folder)


for index in range(1400, len(points), 10):
    plt.subplot(212)
    print points[index].shape
    print points[index]
    #if len(points[index])>1:
    plt.scatter(points[index][:,0],points[index][:,1] )
    plt.subplot(211)
    plt.plot(cl[index][:,0], cl[index][:,1])
    plt.show()