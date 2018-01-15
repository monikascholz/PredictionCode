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
RMatrix = np.loadtxt(folder+'../'+'Rotationmatrix.dat')
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
# coordinate segments
diffVec = np.diff(cl, axis=1)
# calculate tangential vectors
wcNew = np.unwrap(np.arctan2(-diffVec[:,:,1], diffVec[:,:,0]))
#################these things are needed for reconstruction
# get mean angle
meanAngle = np.mean(wcNew, axis=1)
# get segment lengths
lengths = np.sqrt(diffVec[:,:,1]**2+diffVec[:,:,0]**2)
print cl.shape
# get overall alignment in space
tmpO =cl[:,1]-cl[:,0]
orientation = np.arctan2(tmpO[:,1], tmpO[:,0])
# reference point to start with
refPoint = cl[:,0]
# calculate mean subtracted tangent angles
wcNew = wcNew-meanAngle[:, np.newaxis]
#plt.plot(wcNew[index])
#plt.plot(wc[index])
#plt.show()
# project onto Eigenworms
pcsNew = np.dot(eigenworms,wcNew.T)
# getting the same eigenworms as are written in heatmap data
#
#for i in range(3):
#    plt.subplot(3,1,i+1)
#    plt.plot(ep[:,i])
#    plt.plot(pcsNew[i], alpha=0.5)
#    plt.plot(pcs[:,i], alpha=0.5)
#plt.show()

plt.plot(eigenworms.T)
plt.show()


# now we recreate the worm
wcR = np.dot(pcsNew.T, eigenworms) + meanAngle[:, np.newaxis] 
wcR = -wcR
print wcR[index,0]*180/np.pi, orientation[index]*180/np.pi
#wcR = wcR -wcR[:,0][:, np.newaxis]
print wcR[index,0]*180/np.pi, orientation[index]*180/np.pi#
#wcR = wcR +orientation[:, np.newaxis]
print wcR[index,0]*180/np.pi, orientation[index]*180/np.pi
#wcR += orientation[:, np.newaxis] -wcR[:,0][:, np.newaxis]
#wcR[:,0] = orientation
#print wcR[index]
#plt.plot(wcR[index])
#plt.plot(wc[index])
#plt.show()
# recreate tangent vectors with correct length
tVecs = np.stack([lengths*np.cos(wcR), lengths*np.sin(wcR)], axis=2)
print tVecs.shape
print cl.shape
# start at same point as original CL
clApprox = refPoint[:, np.newaxis] + np.cumsum(tVecs, axis=1)


plt.plot(cl[index][:,0], cl[index][:,1], 'o')

plt.plot(clApprox[index][:,0], clApprox[index][:,1], 'o')
plt.show()

for i,index in enumerate(range(100,500,100)):

    plt.subplot(2,2,i+1)
    plt.plot(cl[index][:,0], cl[index][:,1])
    
    plt.plot(clApprox[index][:,0], clApprox[index][:,1])
plt.show()
    
    
    
    
    
