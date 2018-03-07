# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:21:23 2018
fit GCamp6s response to a model.
@author: monika
"""
import numpy as np
import matplotlib.pylab as plt
import scipy.interpolate
from scipy.signal import medfilt, deconvolve
from scipy.special import erf
from scipy.optimize import curve_fit
#deconvolve with GCamp6s response digitized from Nature volume 499, pages 295–300 (18 July 2013)
#doi:10.1038/nature12354"""
# load gcamp response - x is in seconds
xdata, ydata = np.loadtxt('GCamp6s_envelope.csv',delimiter=',', unpack=True)
ydata = ydata[xdata>0]
ydata -= ydata[0]
xdata = xdata[xdata>0]
# fit function
def fitfun(x,A,m, tau1, s, tau2):
    return A*erf((x-m)/s)*np.exp(-x/tau1)#*np.exp(-x/tau2)
    
# define a fit function
plt.plot(xdata, ydata, 'b-', label='data')
p0=(0.5,0.25,1,0.3, 3)
plt.plot(xdata, fitfun(xdata, *p0))

popt, pcov = curve_fit(fitfun, xdata, ydata, p0)
print popt
plt.plot(xdata, fitfun(xdata, *popt), 'r-',
         label='fit: A=%5.3f, m=%5.3f, tau1=%5.3f, s=%5.3f,tau2=%5.3f' % tuple(popt))
plt.legend()
plt.show()