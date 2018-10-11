# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:11:50 2018
test ratiometric versus ICA for some toy data.
@author: monika
"""
import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import FastICA
from stylesheet import *
#####################################
fig = plt.figure('FigICA', figsize=(9.5, 6.5))
letters = ['A', 'B', 'C', 'D']
x0 = 0
locations = [(x0,0.95),  (0.5,0.95), (0,0.5),  (0.5,0.5)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
######################################
x = np.arange(100)
#R = 100-np.arange(100)+np.random.rand(100)
#G = 1-0.1*np.arange(100)+1*np.random.rand(100)
R = 2*np.exp(-x/50)+np.random.rand(100)
G = 2*np.exp(-x/50)+np.random.rand(100)

A = 0.25
R = 1+A*np.random.normal(loc=0, scale=1, size=100) #+ 5*np.exp(-x/10)
G = 1+ A*np.random.normal(loc=0, scale=1, size=100)#+ 5*np.exp(-x/20)
######################################
S = np.ones(len(x))
Bg = np.ones(len(x))
S[15:20] +=2.5
S[30:40] +=2.5
S[55:60] +=4.5
S[75:80] +=8.5

Bg[30:40] +=2.5

# add signal and background
R += Bg
G += S

# true signal
S = S/Bg
S0 = np.percentile(S/Bg, [20])
S = np.divide(S-S0,np.abs(S0))

# ICA signal
ica = FastICA(n_components=2)
signal = np.vstack([R,G])
comp = ica.fit_transform(signal.T)
index = np.argmax([np.abs(np.corrcoef(s, G/R)[0][1]) for s in comp.T])
factor = np.sign(np.corrcoef(comp[:,index], G)[0][1])
comp *=factor

I0 =np.percentile(comp, [20], axis=0)
ICA = np.divide(comp-I0,np.abs(I0))


# ratiometric signal
G0, R0 = np.percentile(G, [20]), np.percentile(R, [20])

tmpRatio = (G/G0)/(R/R0)
Ratio0 = np.percentile(tmpRatio, [20])
Ratio = np.divide(tmpRatio-Ratio0,np.abs(Ratio0))



plt.subplot(221)
plt.ylabel('Raw intensity')
plt.plot(R, R1, label="Red indicator")
plt.plot(G, B1, label="Green indicator")
plt.subplot(222)

plt.ylabel(r"$\Delta R/R_0$")
plt.plot(Ratio, L0)
plt.plot(S, 'k--',zorder=-1, lw=1.5)
plt.ylim(-1,10)
plt.subplot(223)
plt.ylabel('ICA background (a.u)')
plt.plot(ICA[:,1-index])
plt.ylim(-1,10)
plt.subplot(224)
plt.ylabel(r"$\Delta I/I_0$")
plt.plot(ICA[:,index], L0)
plt.plot(S, 'k--', zorder=-1, lw=1.5)
plt.ylim(-1,10)
plt.tight_layout()
plt.show()



