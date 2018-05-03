# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:49:04 2018
Recreate an artificial worm from prediction and show side-by-side with original.
@author: monika
"""
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.animation as animation
import numpy as np
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import explained_variance_score

#custom
import dataHandler as dh
import makePlots as mp

mpl.rcParams["savefig.format"] ='pdf'
mpl.rcParams['legend.markerscale'] = 5
mpl.rcParams["axes.labelsize"]=  12
mpl.rcParams["xtick.labelsize"]=  12
mpl.rcParams["ytick.labelsize"]=  12
mpl.rc('font', **{'sans-serif' : 'FiraSans','family' : 'sans-serif'})

def recrWorm(av, turns, thetaTrue, r, show = 1):
    """recalculate eigenworm prefactor from angular velocity and turns."""
    thetaCum = np.cumsum(av)
    # reset theta every minute to real value   
    dt = np.arange(0, len(thetaCum), 180)
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
    if show:
        plt.figure('Real and reconstructed phase angle')
        plt.subplot(221)
        #plt.plot(thetaCum, label = 'reconstructed')
        #plt.plot(thetaTrue, label = 'real')
        plt.scatter(thetaCum,thetaTrue, label = 'reconstructed')
        #plt.plot(thetaTrue, label = 'real')
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
    
    #=============================================================================#
    #                           # data parameters
    #=============================================================================#
    
    dataPars = {'medianWindow':5, # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow':5, # sgauss window for angle velocity derivative. must be odd
            'rotate':True, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5  # gauss window for red and green channel
            }
    folder = "AML32_moving/"
    dataLog = "AML32_moving/AML32_moving_datasets.txt"
    outLoc = "AML32_moving/Analysis/Results.hdf5"
    
    #=============================================================================#
    #                           # load eigenworms
    #=============================================================================#
    eigenworms = dh.loadEigenBasis(filename='utility/Eigenworms.dat', nComp=3, new = True)    
    #=============================================================================#
    # load datasets from hdf5/matlab in a dictionary
    #=============================================================================#
    dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars, nDatasets = 2)
    keyList = np.sort(dataSets.keys())
    # pick one dataset
    key = keyList[1]    
    data = dataSets[key]
    pc1, pc2, pc3, avTrue, thetaTrue = data['Behavior']['Eigenworm1'],data['Behavior']['Eigenworm2'],\
                            data['Behavior']['Eigenworm3'],  data['Behavior']['AngleVelocity'],  data['Behavior']['Theta']
    pcs = np.vstack([pc3,pc2, pc1])
    # actual centerline
    cl= data['CL']     
    #=============================================================================#
    # for debugging recreate an existing, approximated shape from 3 eigenworms
    #=============================================================================#    
    pcsNew, meanAngle, lengths, refPoint = dh.calculateEigenwormsFromCL(cl, eigenworms)
    pc3New, pc2New, pc1New = pcsNew
    cl = dh.calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint)
    # transform eigenworms exactly the same way. Otherwise we get some artefacts from nans
    #_,_,_, av, theta = dh.transformEigenworms(pcsNew, dataPars)
    r = (pcsNew[2]**2+pcsNew[1]**2)
    #=============================================================================#
    # here we reconstruct from the true angular velocity to check the math. This is smoothed, so we need to compare with this version
    #=============================================================================#
    xt, yt, zt = recrWorm(avTrue, pc3, thetaTrue, r=r)
    pcsR = np.vstack([zt,yt, xt])
    clApprox = dh.calculateCLfromEW(pcsR, eigenworms, meanAngle, lengths, refPoint)
    #=============================================================================#
    # load predicted worm
    #=============================================================================#
    resultDict = dh.loadDictFromHDF(outLoc)
    results = resultDict[key]
    avP = results['ElasticNet']['AngleVelocity']['output'][:len(pcs[0])]
    tP = results['ElasticNet']['Eigenworm3']['output'][:len(pcs[0])]
    print 'R2'
    print results['ElasticNet']['AngleVelocity']['score'],results['ElasticNet']['AngleVelocity']['scorepredicted']
    print results['ElasticNet']['Eigenworm3']['score'],results['ElasticNet']['Eigenworm3']['scorepredicted']
    #=============================================================================#    
    # reconstruct worm from predicted angular velocity.
    #=============================================================================#
    xP, yP, zP = recrWorm(avP, tP, thetaTrue,r=r)
    pcsP = np.vstack([zP,yP, xP])
    clPred = dh.calculateCLfromEW(pcsP, eigenworms, meanAngle, lengths, refPoint)
    # center around midpoint
    originalCMS = np.tile(np.mean(cl, axis=1)[:,np.newaxis,:], (1,99,1))
    clApprox -= originalCMS
    cl -= originalCMS
    clPred -= originalCMS# +(0,100)

    if show:
        
        # animate centerlines
        fig = plt.figure('wiggly centerlines')
        ax0 = fig.add_subplot(311)
        plt.plot(avTrue, label = 'True', color=mp.colorBeh['AngleVelocity'])
        plt.plot(avP, label = 'Predicted', color=mp.colorPred['AngleVelocity'])
        plt.xlabel('Frames')
        plt.ylabel('Wave velocity')
        plt.legend()
        plt.subplot(312)
        plt.plot(pc3, color=mp.colorBeh['Eigenworm3'])
        plt.plot(tP, color=mp.colorPred['Eigenworm3'])
        plt.xlabel('Frames')
        plt.ylabel('Wave velocity')
        ax = fig.add_subplot(313, adjustable='box', aspect=0.66)

        ax.set_ylim(-400, 400)
        ax.set_xlim(-400, 800)
        
        mp.make_animation3(fig, ax, data1= clPred, data2=clApprox +(500,0) , frames = results['Training']['AngleVelocity']['Test'], save=True)
        #mp.make_animation2(fig, ax, data1= clApprox +(500,0), data2=clApprox +(500,0) , frames = np.arange(120, 1000), color=mp.UCred[0])
        plt.show()
        
        # show overlay of eigenworms, correlations etc.
        plt.figure('True reconstruction relative to original centerlines')
        for i in range(3):
            plt.subplot(3,1,i+1)
           
            #plt.plot(pcsNew[i], alpha=0.5, label='centerline projected')
            plt.plot(pcs[i], alpha=0.5, label='dataHandler')
            plt.plot(pcsR[i], alpha=0.5, label='true reconstructed')
            plt.legend()
            plt.ylabel('Projection on Eigenworm {}'.format(3-i))
            plt.xlabel('Time (Volumes)')
        plt.tight_layout()
        plt.show()       
        #example centerlines
        i=99
        plt.plot(clApprox[i,:,0],clApprox[i,:,1]+300*i, '0.5', label = 'Approximated centerline by 4 Eigenworms')
        plt.plot(cl[i,:,0],cl[i,:,1]+300*i, 'r', label = 'Real Centerline')
        plt.plot(clPred[i,:,0],clPred[i,:,1]+300*i, 'b', label = 'Predicted Centerline')
        for i in range(100, 200, 5):
            plt.plot(clApprox[i,:,0],clApprox[i,:,1]+300*i, '0.5')
            plt.plot(cl[i,:,0],cl[i,:,1]+300*i, 'r')
            plt.plot(clPred[i,:,0],clPred[i,:,1]+300*i, 'b')
        plt.legend()
        plt.tight_layout()
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
        plt.tight_layout()
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

        
main()