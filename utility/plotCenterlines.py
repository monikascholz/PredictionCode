# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 09:58:31 2017
plot centerlines.
@author: monika
"""
import matplotlib.pylab as plt
import dataHandler as dh
import numpy as np
from skimage.transform import resize

def calculateEigenwormsFromCL(cl, eigenworms):
    """takes (x,y) pairs from centerlines and returns eigenworm coefficients."""
    # coordinate segments
    diffVec = np.diff(cl, axis=1)
    # calculate tangential vectors
    wcNew = np.unwrap(np.arctan2(-diffVec[:,:,1], diffVec[:,:,0]))
    #################these things are needed for reconstruction
    # get mean angle
    meanAngle = np.mean(wcNew, axis=1)
    # get segment lengths
    lengths = np.sqrt(diffVec[:,:,1]**2+diffVec[:,:,0]**2)
    # get overall alignment in space
    # reference point to start with
    refPoint = cl[:,0]
    # calculate mean subtracted tangent angles
    wcNew = wcNew-meanAngle[:, np.newaxis]
    # project onto Eigenworms
    pcsNew = np.dot(eigenworms,wcNew.T)
    return pcsNew, meanAngle, lengths, refPoint

def calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint):
    """takes eigenworms and a few reference parameters to recreate centerline."""
    # now we recreate the worm
    wcR = np.dot(pcsNew.T, eigenworms) + meanAngle[:, np.newaxis] 
    # recreate tangent vectors with correct length
    tVecs = np.stack([lengths*np.cos(wcR), -lengths*np.sin(wcR)], axis=2)
    # start at same point as original CL
    clApprox = refPoint[:, np.newaxis] + np.cumsum(tVecs, axis=1)
    return clApprox

#folder = "AML32_moving/{}_MS/"
#dataLog = "AML32_moving/AML32_datasets.txt"
#dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder)
##  
#keyList = np.sort(dataSets.keys())
# load centerline data eigenproj
dataPars = {'medianWindow':1, # smooth eigenworms with median filter of that size, must be odd
            'savGolayWindow':13, # savitzky-golay window for angle velocity derivative. must be odd
            'rotate':False, # rotate Eigenworms using previously calculated rotation matrix
            'savGolayWindowGCamp': 5 # savitzky-golay window for red and green channel
            }

folder = "../BrainScanner20170613_134800_MS/"
ewfile = "eigenWorms.mat"
#RMatrix = np.loadtxt(folder+'../'+'Rotationmatrix.dat')
points = dh.loadPoints(folder)
#print points
cl, wc, ep = dh.loadCenterlines(folder)
#cl = dh.loadCenterlines(folder, wormcentered=0)

data = dh.loadData(folder, dataPars)
eigenworms = dh.loadEigenBasis(ewfile)[:3]
print eigenworms.shape
eigenworms = resize(eigenworms, (3,99))

pc1,pc2,pc3 = [data['Behavior']['Eigenworm1'],data['Behavior']['Eigenworm2'],data['Behavior']['Eigenworm3']]
# reverse rotate the Eigenworms
#pcs = np.array(np.dot(-RMatrix, np.vstack([pc1, pc2, pc3]))).T 
# switch pc1 and two--they otherwise are switched everywhere :-(
#IMPORTANT TODO: Change to correct this
pcs = np.vstack([pc3, pc1, pc2]).T
print 'prefactors', pcs.shape
print 'Eigenworms', eigenworms.shape


index = 1000

# getting the same eigenworms as are written in heatmap data
pcsNew, meanAngle, lengths, refPoint = calculateEigenwormsFromCL(cl, eigenworms)

#
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(ep[:,i])
    plt.plot(pcsNew[i], alpha=0.5)
    plt.plot(pcs[:,i], alpha=0.5)
plt.show()

plt.plot(eigenworms.T)
plt.show()


clApprox = calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint)


plt.plot(cl[index][:,0], cl[index][:,1], 'o')

plt.plot(clApprox[index][:,0], clApprox[index][:,1], 'o')
plt.show()

for i,index in enumerate(range(100,500,100)):

    plt.subplot(2,2,i+1)
    plt.plot(cl[index][:,0], cl[index][:,1])
    
    plt.plot(clApprox[index][:,0], clApprox[index][:,1])
plt.show()
    
    
    
    
    