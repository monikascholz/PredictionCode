# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:49:04 2018
Recreate an artificial worm from prediction and show side-by-side with original.
@author: monika
"""
import matplotlib as mpl
import matplotlib.pylab as plt
import dataHandler as dh
import numpy as np
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import explained_variance_score

mpl.rcParams["savefig.format"] ='pdf'
mpl.rcParams['legend.markerscale'] = 5
mpl.rcParams["axes.labelsize"]=  12
mpl.rcParams["xtick.labelsize"]=  12
mpl.rcParams["ytick.labelsize"]=  12
mpl.rc('font', **{'sans-serif' : 'FiraSans','family' : 'sans-serif'})

def recrWorm(av, turns, thetaTrue, r):
    """recalculate eigenworm prefactor from angular velocity and turns."""
    thetaCum = np.cumsum(av)
    
    # reset theta every minute to real value   
    dt = np.arange(0, len(thetaCum), 60)
    for tt in dt:    
        thetaCum[tt:] -= -thetaTrue[tt]+thetaCum[tt]
#    thetaCum -= thetaCum[50]-thetaTrue[50] 
#    thetaCum -= np.mean(thetaCum)- np.mean(thetaTrue)
    #tt = 0
    #thetaCum[tt:] -= -thetaTrue[tt]+thetaCum[tt]
    # recalculate the phase angle from cumulative phase
    theta = np.mod(thetaCum, 2*np.pi)#-np.pi
    sign = np.ones(len(theta))
    sign[np.where(np.abs(theta-np.pi)>np.pi/2.)] = -1
    # do the same for the true angle
    thetaD = np.mod(thetaTrue, 2*np.pi)
    thetaDS = np.where(np.abs(thetaD-np.pi)>np.pi/2., -thetaD, thetaD)
    
    plt.figure('Real and reconstructed phase angle')
    plt.subplot(221)
    plt.plot(thetaCum, label = 'reconstructed')
    plt.plot(thetaTrue, label = 'real')
    plt.ylabel('Accumulated phase angle')
    plt.legend()
    plt.subplot(222)
    plt.plot(thetaCum-thetaTrue, label = 'residuals')
    plt.plot(np.cumsum(thetaCum-thetaTrue), label = 'cumulative residuals')
    plt.ylabel('Phase difference')
    plt.legend()
    plt.subplot(223)
    plt.plot(theta, label = 'reconstructed')
    plt.plot(thetaD, label ='real')#-np.pi)
    plt.ylabel('Phase angle (rad)')
    plt.xlabel('Time (Volumes)')
    plt.legend()
    plt.subplot(224)
    plt.scatter(sign*theta,thetaDS,s=1, alpha=0.1)
    plt.ylabel('Phase angle (rad)')
    plt.xlabel('reconstructed Phase angle (rad)')
    plt.tight_layout()
    plt.show()
    # recalculate eigenworms
    x = -np.sqrt(r)*np.tan(sign*theta)/np.sqrt((np.tan(sign*theta)**2+1))
    y = -sign*np.sqrt(r-x**2)
    return x,y, turns
    
    
def main():    
    show = 1
    # data parameters
    dataPars = {'medianWindow':11, # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow':11, # sgauss window for angle velocity derivative. must be odd
            'rotate':True, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5  # gauss window for red and green channel
            }
    folder = "AML32_moving/{}_MS/"
    dataLog = "AML32_moving/AML32_datasets.txt"
    outLoc = "AML32_moving/Analysis/Results.hdf5"
    
    # load eigenworms
    eigenworms = dh.loadEigenBasis(filename='', nComp=3, new = True)    
    
    # load datasets from hdf5 in a dictionary
    dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars, nDatasets = 1)
    keyList = np.sort(dataSets.keys())
    key = keyList[0]
    # for debugging recreate an existing, approximated shape. These are now our new, cool eigenworms with correct shapes
    data = dataSets[key]
    pc1, pc2, pc3, avTrue, thetaTrue = data['Behavior']['Eigenworm1'],data['Behavior']['Eigenworm2'],\
                            data['Behavior']['Eigenworm3'],  data['Behavior']['AngleVelocity'],  data['Behavior']['Theta']
    #
    pcs = np.vstack([pc3,pc2, pc1])
    # from centerline
    cl= data['CL']
    pcsNew, meanAngle, lengths, refPoint = dh.calculateEigenwormsFromCL(cl, eigenworms)
    pc3New, pc2New, pc1New = pcsNew
    # transform eigenworms exactly the same way. Otherwise we get some artefacts from nans
    #_,_,_, av, theta = dh.transformEigenworms(pcsNew, dataPars)
    r = (pcsNew[2]**2+pcsNew[1]**2)
    # reconstruct worm from angular velocity. sanity check
    #theta = theta[data['goodVolumes']]
    #r = r[data['goodVolumes']]
    #av = av[data['goodVolumes']]
    xt, yt, zt = recrWorm(avTrue, pc3New, thetaTrue, r=r)
    pcsR = [zt,yt, xt]
    
    if show:
        plt.figure('True reconstruction relative to original centerlines')
        for i in range(3):
            plt.subplot(3,1,i+1)
           
            plt.plot(pcsNew[i], alpha=0.5, label='centerline projected')
            plt.plot(pcs[i], alpha=0.5, label='dataHandler')
            plt.plot(pcsR[i], alpha=0.5, label='true reconstructed')
            plt.legend()
            plt.ylabel('Projection on Eigenworm {}'.format(3-i))
            plt.xlabel('Time (Volumes)')
        plt.show()       
        #plt.subplot(211)
#        plt.plot(theta-thetaTrue)
#        plt.plot(thetaTrue)
#        plt.subplot(212)
#        plt.plot(avTrue-av, label = 'from dataHandler')
#        plt.plot(av, label = 'from pcs')
#        plt.ylabel('Angle velocity')
#        plt.show()
        
        # show true reconstruction
        plt.figure('True reconstruction relative to original centerlines')
        plt.subplot(311)
        plt.plot(xt,label = 'True reconstructed', alpha=0.5)    
        plt.plot(pcs[2], label = 'True projected', alpha=0.5)  
        plt.ylabel('Projection on Eigenworm 1')
        plt.xlabel('Time (Volumes)')
        plt.subplot(312)
         
        plt.plot(yt,label = 'True reconstructed', alpha=0.5)    
        plt.plot(pcs[1], label = 'True projected', alpha=0.5) 
        plt.ylabel('Projection on Eigenworm 2')
        plt.xlabel('Time (Volumes)')
        plt.subplot(313)
        
        plt.plot(zt, label = 'True reconstructed', alpha=0.5)    
        plt.plot(pcs[0], label = 'True projected', alpha=0.5)
        plt.ylabel('Projection on Eigenworm 3')
        plt.xlabel('Time (Volumes)')
        plt.legend()
        plt.show()
        
        # 
        cut = 1700
        sz = 3
        a = 0.5
        plt.figure(figsize=(12,7))
        plt.subplot(231)
        plt.scatter(yt[:cut], xt[:cut], label = 'True reconstructed', alpha=a, s = sz)
        plt.scatter(pcs[1][:cut], pcs[2][:cut], label = 'True projected', alpha=a, s = sz)
        plt.ylabel('Projection on Eigenworm 1')
        plt.xlabel('Projection on Eigenworm 2')
        plt.legend()
        plt.subplot(234)
        plt.scatter(yt[cut:], xt[cut:], label = 'True reconstructed', alpha=a, s = sz)
        plt.scatter(pcs[1][cut:], pcs[2][cut:], label = 'True projected', alpha=a, s = sz)
        plt.legend()
        plt.ylabel('Projection on Eigenworm 1')
        plt.xlabel('Projection on Eigenworm 2')
        
        plt.subplot(232)
        plt.scatter(xt[:cut], pcs[2][:cut], label = 'First part R^2 = {:.2f}'.format(explained_variance_score(xt[:cut], pcs[2][:cut])), alpha=a, s = sz, color ='C2' )
        plt.ylabel('Projection on Eigenworm 1')
        plt.xlabel('Reconstruction')
        plt.legend()
        plt.subplot(235)
        plt.scatter(xt[cut:], pcs[2][cut:], label = 'Second part R^2 = {:.2f}'.format(explained_variance_score(xt[cut:], pcs[2][cut:])), alpha=a, s = sz, color ='C2' )
        plt.legend()
        plt.ylabel('Projection on Eigenworm 1')
        plt.xlabel('Reconstruction')
        plt.subplot(233)
        plt.scatter(yt[:cut], pcs[1][:cut], label = 'First part R^2 = {:.2f}'.format(explained_variance_score(yt[:cut], pcs[1][:cut])), alpha=a, s = sz, color ='C3' )
        plt.ylabel('Projection on Eigenworm 2')
        plt.xlabel('Reconstruction')
        plt.legend()
        plt.subplot(236)
        plt.scatter(yt[cut:], pcs[1][cut:], label = 'Second part R^2 = {:.2f}'.format(explained_variance_score(yt[cut:], pcs[1][cut:])), alpha=a, s = sz, color ='C3' )
        plt.legend()
        plt.ylabel('Projection on \n Eigenworm 2')
        plt.xlabel('Reconstruction')
        plt.tight_layout()
        plt.show()
        
    # load predicted worm
    resultDict = dh.loadDictFromHDF(outLoc)
    results = resultDict[key]['LASSO']
    avP = results['AngleVelocity']['output'][:len(pcs[0])]
    tP = results['Eigenworm3']['output'][:len(pcs[0])]
    print 'R2'
    print results['AngleVelocity']['score'],results['AngleVelocity']['scorepredicted']
    print results['Eigenworm3']['score'],results['Eigenworm3']['scorepredicted']
    # reconstruct worm from predicted angular velocity.
    #r = xt**2 + yt**2
    xP, yP, zP = recrWorm(avP, tP, thetaTrue,r=r)
    
    plt.figure('Predicted velocity and turns')
    plt.subplot(211)
    plt.plot(avTrue, label = 'True')
    plt.plot(avP, label = 'Predicted')
    plt.legend()
    plt.subplot(212)
    plt.plot(pc3)
    plt.plot(tP)
    plt.tight_layout()
    plt.show()
    
    # show predicted reconstruction
    plt.figure('Predicted Eigenworm reconstruction')
    plt.subplot(311)
    plt.plot(xP,label = 'Prediction reconstructed', alpha=0.5)    
    plt.plot(xt, label = 'True projected', alpha=0.5)  
    plt.ylabel('Projection on Eigenworm 1')
    plt.xlabel('Time (Volumes)')
    plt.subplot(312)
     
    plt.plot(yP,label = 'Prediction reconstructed', alpha=0.5)    
    plt.plot(yt, label = 'True projected', alpha=0.5) 
    plt.ylabel('Projection on Eigenworm 2')
    plt.xlabel('Time (Volumes)')
    plt.subplot(313)
    
    plt.plot(zP, label = 'Prediction reconstructed', alpha=0.5)    
    plt.plot(zt, label = 'True projected', alpha=0.5)
    plt.ylabel('Projection on Eigenworm 3')
    plt.xlabel('Time (Volumes)')
    plt.legend()
    
    
    # 
    cut = 1700
    sz = 3
    a = 0.5
    plt.figure(figsize=(12,7))
    plt.subplot(231)
    plt.scatter(yt[:cut], xt[:cut], label = 'True reconstructed', alpha=a, s = sz)
    plt.scatter(yP[:cut], xP[:cut], label = 'Predicted', alpha=a, s = sz)
    plt.ylabel('Projection on Eigenworm 1')
    plt.xlabel('Projection on Eigenworm 2')
    plt.legend()
    plt.subplot(234)
    plt.scatter(yt[cut:], xt[cut:], label = 'True reconstructed', alpha=a, s = sz)
    plt.scatter(yP[cut:],xP[cut:], label = 'Predicted', alpha=a, s = sz)
    plt.legend()
    plt.ylabel('Projection on Eigenworm 1')
    plt.xlabel('Projection on Eigenworm 2')
    
    plt.subplot(232)
    plt.scatter(xt[:cut], xP[:cut], label = 'First part R^2 = {:.2f}'.format(explained_variance_score(xt[:cut], xP[:cut])), alpha=a, s = sz, color ='C2' )
    plt.legend()    
    plt.ylabel('Predicted')
    plt.xlabel('Reconstruction')
    plt.subplot(235)
    plt.scatter(xt[cut:], xP[cut:], label = 'Second part R^2 = {:.2f}'.format(explained_variance_score(xt[cut:], xP[cut:])), alpha=a, s = sz, color ='C2' )
    plt.legend()  
    plt.ylabel('Predicted')
    plt.xlabel('Reconstruction')
    plt.subplot(233)
    plt.scatter(yt[:cut], yP[:cut], label = 'First part R^2 = {:.2f}'.format(explained_variance_score(yt[:cut], yP[:cut])), alpha=a, s = sz, color ='C3' )
    plt.legend()   
    plt.ylabel('Predicted')
    plt.xlabel('Reconstruction')
    plt.legend()
    plt.subplot(236)
    plt.scatter(yt[cut:], yP[cut:], label = 'Second part R^2 = {:.2f}'.format(explained_variance_score(yt[cut:],yP[cut:])), alpha=a, s = sz, color ='C3' )
    plt.legend()
    plt.ylabel('Predicted')
    plt.xlabel('Reconstruction')
    plt.tight_layout()
    plt.show()
#    for key in resultDict.keys()[:1]:
#        # get data handler eigenworms
#        data = dh.loadData(folder.format(key), dataPars)
#    
#        pc1, pc2, pc3, avTrue, thetaTrue = data['Behavior']['Eigenworm1'],data['Behavior']['Eigenworm2'],\
#                            data['Behavior']['Eigenworm3'],  data['Behavior']['AngleVelocity'],  data['Behavior']['Theta']
#        pcdh = np.vstack([pc1,pc2, pc3])
#        # for now get one LASSO worm
#        cl= data['CL']# dh.loadCenterlines(folder.format(key))
#        # get rotation matrix
#        RMatrix = np.loadtxt(folder.format(key)+'../'+'Rotationmatrix.dat')
#        # get reference points from CL we will need for recreation
#        # getting the same eigenworms as are written in heatmap data
#        pcsNew, meanAngle, lengths, refPoint = dh.calculateEigenwormsFromCL(cl, eigenworms)
#        pcs = np.array(np.dot(RMatrix, np.vstack([pcsNew[0],pcsNew[1], pcsNew[2]])))
        
        
        
        #
#        print pcdh.shape, pcs.shape, pcsNew.shape, meanAngle.shape
#    #    plt.plot(np.unwrap(np.arctan2(pcs[1], pcs[0])))
#    #    plt.plot(thetaTrue)
#    #    plt.show()
#        for i in range(3):
#            plt.subplot(3,1,i+1)
#            plt.plot(pcdh[i], label='from pipeline, rotated')
#            plt.plot(pcsNew[i], alpha=0.5, label='centerline projected')
#            plt.plot(pcs[i], alpha=0.5, label='centerline projected, rotated, reordered')
#        plt.legend()
#        plt.show()       
#        
#        # calculate predicted CL
#        results = resultDict[key]['LASSO']
#        av = results['AngleVelocity']['output'][:len(pcs[0])]
#        t = results['Eigenworm3']['output'][:len(pcs[0])]
#        # from real worm
#        #thetaTrue = np.unwrap(np.arctan2(pcs[0], pcs[1]))
#        
#        r = np.mean(pcs[0]**2+pcs[1]**2)
#        # predicted worm
#        print 'predicted worm'
#        x,y,z = recrWorm(av, t, thetaTrue, r)
#        # recreated true worm
#        #avt =  gaussian_filter1d(thetaTrue, dataPars['savGolayWindow'], order=1)
#        plt.ylabel('angular velocity')
#        plt.plot(av, label = 'from prediction')
#        plt.plot(avTrue, label= 'from datahandler')
#        plt.legend()
#        plt.show()
#        
#        xt, yt, zt = recrWorm(avTrue, pc3, thetaTrue, r)
#        
#        
#        plt.subplot(311)
#        plt.plot(x, label = 'predicted', alpha=0.5)    
#        plt.plot(xt,label = 'True reconstructed', alpha=0.5)    
#        plt.plot(pcs[0], label = 'True projected', alpha=0.5)    
#        plt.subplot(312)
#        plt.plot(y, label = 'predicted', alpha=0.5)    
#        plt.plot(yt,label = 'True reconstructed', alpha=0.5)    
#        plt.plot(pcs[1], label = 'True projected', alpha=0.5)    
#        plt.subplot(313)
#        plt.plot(z, label = 'predicted', alpha=0.5)    
#        plt.plot(zt, label = 'True reconstructed', alpha=0.5)    
#        plt.plot(pcs[2], label = 'True projected', alpha=0.5)
#        plt.legend()
#        plt.show()
#        plt.subplot(311)
#          
#        
#        
#        print RMatrix
#        # rotate back and reorder
#        synPCs = np.array(np.dot(RMatrix.T, np.vstack([x,y,z])))[::-1]
#        synEW = dh.calculateCLfromEW(synPCs, eigenworms, meanAngle, lengths, refPoint)
#        print x.shape, xt.shape
#        truePCs = np.array(np.dot(RMatrix, np.vstack([xt,yt,zt])))[::-1]
#        print RMatrix
#        trueEW = dh.calculateCLfromEW(truePCs, eigenworms, meanAngle, lengths, refPoint)
#        # calculate approximate CL
#        clApprox = dh.calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint)
#        # plot actual CL and approximate CL
#        for i,index in enumerate(range(100,1600,100)):
#        
#            plt.subplot(4,4,i+1)
#            #plt.plot(cl[index][:,0], cl[index][:,1])
#            
#            plt.plot(clApprox[index][:,0], clApprox[index][:,1], label='centerline')
#            plt.plot(synEW[index][:,0], synEW[index][:,1], label='predicted worm')
#            plt.plot(trueEW[index][:,0], trueEW[index][:,1], label = 'true reconstructed worm')
#        plt.show()
        
main()