# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:49:04 2018
Recreate an artificial worm from prediction and show side-by-side with original.
@author: monika
"""

import matplotlib.pylab as plt
import dataHandler as dh
import numpy as np
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter1d

def recrWorm(av, turns, thetaTrue, r):
    """recalculate eigenworm prefactor from angular velocity and turns."""
    theta = np.cumsum(av)
    # reset theta every minute to real value   
    dt = np.arange(0, len(theta), 60*10)
    for tt in dt:    
        theta[tt:] -= -thetaTrue[tt]+theta[tt]
    plt.plot(theta)
    plt.plot(thetaTrue)
    plt.show()
    r = pcs[0]**2+pcs[1]**2
    theta = np.mod(theta, 2*np.pi)#-np.pi
    plt.plot(theta)
    plt.plot(np.mod(thetaTrue, 2*np.pi))#-np.pi)
    plt.show()
    # recalculate eigenworms
    sign = np.ones(len(theta))
    sign[np.where(np.abs(theta-np.pi)<np.pi/2.)] = -1

    y = np.sqrt(r)*np.tan(sign*theta)/np.sqrt((np.tan(theta)**2+1))
    x = sign*np.sqrt(r-y**2)
    return x,y, turns
    
# load old eigenworms that we use in the pipeline
ewfile = "eigenWorms.mat"
# ncomponents controls how many we use
nComp = 3
eigenworms = dh.loadEigenBasis(ewfile)[:nComp]
eigenworms = resize(eigenworms, (nComp,99))


dataPars = {'medianWindow':1, # smooth eigenworms with median filter of that size, must be odd
            'savGolayWindow':7, # savitzky-golay window for angle velocity derivative. must be odd
            'rotate':True, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5 # savitzky-golay window for red and green channel
            }

folder = "AML32_moving/{}_MS/"
dataLog = "AML32_moving/AML32_datasets.txt"
outLoc = "AML32_moving/Analysis/Results.hdf5"

# load datasets from hdf5 in a dictionary
dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars)
keyList = np.sort(dataSets.keys())
resultDict = dh.loadDictFromHDF(outLoc)
for key in resultDict.keys()[:1]:
    # get data handler eigenworms
    data = dh.loadData(folder.format(key), dataPars)


    pc1, pc2, pc3, avTrue, thetaTrue = data['Behavior']['Eigenworm1'],data['Behavior']['Eigenworm2'],\
                        data['Behavior']['Eigenworm3'],  data['Behavior']['AngleVelocity'],  data['Behavior']['Theta']
    pcdh = np.vstack([pc1,pc2, pc3])
    # for now get one LASSO worm
    cl, _ = dh.loadCenterlines(folder.format(key))
    # get rotation matrix
    RMatrix = np.loadtxt(folder.format(key)+'../'+'Rotationmatrix.dat')
    # get reference points from CL we will need for recreation
    # getting the same eigenworms as are written in heatmap data
    pcsNew, meanAngle, lengths, refPoint = dh.calculateEigenwormsFromCL(cl, eigenworms)
    
    
     # !!!! ToDO change order of eigenworms! wtf
    pcs = np.array(np.dot(RMatrix, np.vstack([pcsNew[2],pcsNew[1], pcsNew[0]])))
    #
#    plt.plot(np.unwrap(np.arctan2(pcs[1], pcs[0])))
#    plt.plot(thetaTrue)
#    plt.show()
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(pcdh[i], label='from pipeline, rotated')
        plt.plot(pcsNew[i], alpha=0.5, label='centerline projected')
        plt.plot(pcs[i], alpha=0.5, label='centerline projected, rotated, reordered')
    plt.legend()
    plt.show()       
    
    # calculate predicted CL
    results = resultDict[key]['LASSO']
    av = results['AngleVelocity']['output']
    t = results['Eigenworm3']['output']
    # from real worm
    #thetaTrue = np.unwrap(np.arctan2(pcs[0], pcs[1]))
    
    r = np.mean(pcs[0]**2+pcs[1]**2)
    # predicted worm
    print 'predicted worm'
    x,y,z = recrWorm(av, t, thetaTrue, r)
    # recreated true worm
    #avt =  gaussian_filter1d(thetaTrue, dataPars['savGolayWindow'], order=1)
    
    plt.plot(av, label = 'from prediction')
    plt.plot(avTrue, label= 'from datahandler')
    plt.legend()
    plt.show()
    xt, yt, zt = recrWorm(avTrue, pcs[2], thetaTrue, r)
    
    
    plt.subplot(311)
    plt.plot(x, label = 'predicted', alpha=0.5)    
    plt.plot(xt,label = 'True reconstructed', alpha=0.5)    
    plt.plot(pcs[0], label = 'True projected', alpha=0.5)    
    plt.subplot(312)
    plt.plot(y, label = 'predicted', alpha=0.5)    
    plt.plot(yt,label = 'True reconstructed', alpha=0.5)    
    plt.plot(pcs[1], label = 'True projected', alpha=0.5)    
    plt.subplot(313)
    plt.plot(z, label = 'predicted', alpha=0.5)    
    plt.plot(zt, label = 'True reconstructed', alpha=0.5)    
    plt.plot(pcs[2], label = 'True projected', alpha=0.5)
    plt.legend()
    plt.show()
    plt.subplot(311)
      
    plt.plot(x,pcs[0], 'o', alpha=0.1)    
    plt.subplot(312)   
    plt.plot(y, pcs[1],  'o', alpha=0.1)    
    plt.subplot(313)
    plt.plot(z, pcs[2],  'o', alpha=0.1)
    plt.legend()
    plt.show()
    
    print RMatrix
    # rotate back and reorder
    synPCs = np.array(np.dot(RMatrix.T, np.vstack([x,y,z])))[::-1]
    synEW = dh.calculateCLfromEW(synPCs, eigenworms, meanAngle, lengths, refPoint)
    truePCs = np.array(np.dot(RMatrix.T, np.vstack([xt,yt,zt])))[::-1]
    print RMatrix
    trueEW = dh.calculateCLfromEW(truePCs, eigenworms, meanAngle, lengths, refPoint)
    # calculate approximate CL
    clApprox = dh.calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint)
    # plot actual CL and approximate CL
    for i,index in enumerate(range(100,1600,100)):
    
        plt.subplot(4,4,i+1)
        #plt.plot(cl[index][:,0], cl[index][:,1])
        
        plt.plot(clApprox[index][:,0], clApprox[index][:,1], label='centerline')
        plt.plot(synEW[index][:,0], synEW[index][:,1], label='predicted worm')
        plt.plot(trueEW[index][:,0], trueEW[index][:,1], label = 'true reconstructed worm')
    plt.show()