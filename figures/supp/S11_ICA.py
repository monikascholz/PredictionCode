# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:11:50 2018
test ratiometric versus ICA for some toy data.
@author: monika
"""
import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import FastICA
from prediction.stylesheet import *
import matplotlib.gridspec as gridspec
#####################################
fig = plt.figure('FigICA', figsize=(9.5, 7.5))
letters = ['A', 'B', 'C', 'D']
x0 = 0
locations = [(x0,0.95),  (0.5,0.95), (0,0.45),  (0.5,0.45)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
            

gs1 = gridspec.GridSpec(4, 2, height_ratios=[1,4,1,4])
gs1.update(left=0.1, right=0.99, wspace=0.25, bottom = 0.1, top=0.95, hspace=0.2)
ax1 = plt.subplot(gs1[0,0])
ax2 = plt.subplot(gs1[1,0])
ax3 = plt.subplot(gs1[0:2,1])
ax4 = plt.subplot(gs1[3:,0])
ax5 = plt.subplot(gs1[3:,1])
######################################
x = np.arange(100)
#R = 100-np.arange(100)+np.random.rand(100)
#G = 1-0.1*np.arange(100)+1*np.random.rand(100)
R = 2*np.exp(-x/50)+np.random.rand(100)
G = 2*np.exp(-x/50)+np.random.rand(100)

A = 0.25
R = 1+A*np.random.normal(loc=0, scale=1, size=100) #+ 5*np.exp(-x/10)
G = 1+ A*np.random.normal(loc=0, scale=1, size=100)#+ 5*np.exp(-x/50)
######################################
S = np.ones(len(x))
Bg = np.ones(len(x))
S[15:20] +=2.5

S[55:60] +=4.5
S[75:80] +=8.5

# add artefacts
a = 1.5
Bg[30:40] -=a
S[30:40] -=a
#Bg[80:90] -=1.5
#S[80:90] -=1.5

# add signal and background
R += Bg
G += S

# true signal
S[30:40] +=a
#S[80:90] +=1.5
S0 = np.percentile(S, [20])
S = np.divide(S-S0,np.abs(S0))

# ICA signal
ica = FastICA(n_components=2)
signal = np.vstack([R,G])
comp = ica.fit_transform(signal.T)
index = np.argmax([np.abs(np.corrcoef(s, G/R)[0][1]) for s in comp.T])
factor = np.sign(np.corrcoef(comp[:,index], G)[0][1])
comp[:,index]*=factor
factor = np.sign(np.corrcoef(comp[:,1-index], R)[0][1])
comp[:,1-index]*=-factor

I0 = np.percentile(comp, [20], axis=0)
ICA = np.divide(comp-I0,np.abs(I0))


# ratiometric signal
G0, R0 = np.percentile(G, [20]), np.percentile(R, [20])

#tmpRatio = (G/G0)/(R/R0)
tmpRatio=G/R
Ratio0 = np.percentile(tmpRatio, [20])
Ratio = np.divide(tmpRatio-Ratio0,np.abs(Ratio0))


ax1.plot(S, color='k')
ax1.set_title('True signal')
cleanAxes(ax1)

ax2.set_ylabel('Raw intensity')
ax2.plot(R, R1, label="RFP")
ax2.plot(G, 'g', label="GCaMP")
ax2.set_ylim(-1,12)
lh = 0.2
h0=0.6
ax2.annotate('', xy=(0.37, h0), xytext=(0.37, h0+lh),ha="center", va="center",
            arrowprops=dict(facecolor='black', shrink=0.05),xycoords='axes fraction'
            )
ax3.annotate('', xy=(0.37, h0*0.75), xytext=(0.37, (h0+lh)*0.75),ha="center", va="center",
            arrowprops=dict(facecolor='black', shrink=0.05),xycoords='axes fraction'
            )
ax4.annotate('', xy=(0.37, h0), xytext=(0.37, h0+lh),ha="center", va="center",
            arrowprops=dict(facecolor='black', shrink=0.05),xycoords='axes fraction'
            )
ax5.annotate('', xy=(0.37, h0), xytext=(0.37, h0+lh),ha="center", va="center",
            arrowprops=dict(facecolor='black', shrink=0.05),xycoords='axes fraction'
            )


#ax3.annotate('', xy=(35, 5), xytext=(35, 7.5),ha="center", va="center",
#            arrowprops=dict(facecolor='black', shrink=0.05),
#            )
#ax4.annotate('', xy=(35, 5), xytext=(35, 7.5),ha="center", va="center",
#            arrowprops=dict(facecolor='black', shrink=0.05),
#            )
#ax5.annotate('', xy=(35, 5), xytext=(35, 7.5),ha="center", va="center",
#            arrowprops=dict(facecolor='black', shrink=0.05),
#            )
#ax2.set_xlabel("Time (a.u.)")
ax2.legend()

ax3.set_title('Ratiometric')
ax3.set_ylabel(r"$\Delta R/R_0$")
ax3.plot(Ratio, L2)
ax3.plot(S, 'k--',zorder=-1, lw=1.5, label='True signal')
ax3.set_ylim(-2,10)
ax3.set_yticks([0,5,10])
#ax3.set_xlabel("Time (a.u.)")
ax3.legend()

ax4.set_title('ICA background')
ax4.set_ylabel(r"$\Delta I/I_0$")
ax4.plot(ICA[:,1-index])
ax4.set_ylim(-2,10)
ax4.set_xlabel("Time (a.u.)")

ax5.set_title('ICA signal')
ax5.set_ylabel(r"$\Delta I/I_0$")
ax5.plot(ICA[:,index], L0)
ax5.plot(S, 'k--', zorder=-1, lw=1.5)
ax5.set_ylim(-2,10)
ax5.set_xlabel("Time (a.u.)")
ax5.set_yticks([0,5,10])
plt.show()



