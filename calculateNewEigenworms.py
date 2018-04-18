# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:31:40 2018
new eigenbasis from all datasets.
@author: monika
"""
import dataHandler as dh
import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import PCA


# datasets for eigenworm calculation. More than are used for LASSO since 
#it includes animals that had bad neural tracking but ok cl
dataLog = "AML32_moving/AML32_datasets.txt"
pathTemplate = "AML32_moving/{}_MS/"

# load centerline data and concatenate
allCenterlines = []
for lindex, line in enumerate(np.loadtxt(dataLog, dtype=str, ndmin = 2)):
        folder = pathTemplate.format(line[0])
        # get all centerlines, alrady relative angles 'wormcentered' and mean angle subtracted
        allCenterlines.append(dh.loadCenterlines(folder, full = True, wormcentered = True)[0])
        
        
allCenterlines = np.concatenate(allCenterlines)
print 'Datasets included: ', allCenterlines.shape

# run PCA on the whole dataset
nComp = 10
pca = PCA(n_components = nComp)
    
pcs = pca.fit_transform(allCenterlines)
newEigenworms = pca.components_

print 'Explained variance with 4 components: ', np.cumsum(pca.explained_variance_ratio_)[3]

# save data as file
#np.savetxt('Eigenworms.dat', newEigenworms)

# plot old Eigenworms for comparison
oldEigenworms = dh.loadEigenBasis(filename = 'eigenWorms.mat', nComp=4, new=False)

ax4 = plt.subplot(511)
ax4.fill_between(np.arange(0.5,nComp+0.5),pca.explained_variance_ratio_*100, step='post', color='k', alpha=0.75)
#ax4.step(np.arange(1,nComp+1),np.cumsum(results['expVariance'])*100, where = 'pre')
ax4.plot(np.arange(1,nComp+1),np.cumsum(pca.explained_variance_ratio_)*100, 'ko-', lw=1)
ax4.set_ylabel('Explained variance (%)')
ax4.set_yticks([0,25,50,75,100])
ax4.set_xlabel('Number of components')
for i in range(4):
    plt.subplot(5,1,i+2)
    plt.plot(oldEigenworms[i], label = 'old Eigenworms')
    plt.plot(newEigenworms[i], label= 'new Eigenworms')
plt.legend()
plt.show()
# load a centerline dataset
for lindex, line in enumerate(np.loadtxt(dataLog, dtype=str, ndmin = 2)[:1]):
        folder = pathTemplate.format(line[0])
        # get all centerlines, alrady relative angles 'wormcentered' and mean angle subtracted
        cl = dh.loadCenterlines(folder, full = True)[0]
# project a dataset on old and new Eigenworms
pcsOld, meanAngle, lengths, refPoint = dh.calculateEigenwormsFromCL(cl, oldEigenworms)
RMatrix = np.loadtxt(folder+'../'+'Rotationmatrix.dat')
pcs = np.array(np.dot(RMatrix, np.vstack([pcsOld[2],pcsOld[1], pcsOld[0]])))

pcsNew, meanAngle, lengths, refPoint = dh.calculateEigenwormsFromCL(cl, newEigenworms)
pcsNew = pcsNew[[2,1,0]]

plt.subplot(5,1,1)
plt.scatter(pcs[0],pcs[1], label= 'old', alpha=0.05, s=1)
plt.scatter(pcsNew[0]+25,pcsNew[1], label='new', alpha=0.05, s=1)
plt.legend()
for i in range(3):
    plt.subplot(5,1,i+2)
    plt.plot(pcs[i], label='old')
    plt.plot(pcsNew[i], label='new')
    plt.legend()
    # plot new angle velocity
    
theta = np.unwrap(np.arctan2(pcs[0], pcs[1]))
velo = dh.gaussian_filter1d(theta, 15, order=1)
thetaNew = np.unwrap(np.arctan2(pcsNew[0], pcsNew[1]))
veloNew = dh.gaussian_filter1d(thetaNew, 15, order=1)
plt.subplot(5,1,5)
plt.plot(velo)
plt.plot(veloNew)
plt.show()