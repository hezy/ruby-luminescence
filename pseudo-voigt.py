# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:13:38 2019
@author: hezya

https://en.m.wikipedia.org/wiki/Voigt_profile
http://journals.iucr.org/j/issues/1997/04/00/gl0484/gl0484.pdf
http://journals.iucr.org/j/issues/2000/06/00/nt0146/nt0146.pdf
https://www.onlinelibrary.wiley.com/doi/epdf/10.1002/sia.5521
"""

import numpy as np
from scipy.special import wofz
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#import matplotlib as mpl
#from matplotlib.colors import BoundaryNorm
#from matplotlib.ticker import MaxNLocator

def lorentz(x, wL):
    # Lorentz with max=1 and w=FWHM: 
    gamma = wL
    return 1 / (1 + np.square(x/gamma)) 
    

def gauss(x, wG):
    # Gauss with max=1 and w=FWHM
    sigma = wG/np.sqrt(2*np.log(2))
    return np.exp(- x**2 / (2* sigma**2))


def voigt(x, wL, wG):
    gamma = wL
    sigma = wG/np.sqrt(2*np.log(2))
    z = (x + 1j*gamma)/np.sqrt(2)/sigma
    return np.sqrt(2*np.pi) * np.real(wofz(z))/np.sqrt(2*np.pi)/sigma
    # normolized Voigt (integral = 1): c * np.real(wofz((x + 1j*gamma)/(sigma * np.sqrt(2)))) / (sigma * np.sqrt(2*np.pi))
    # for Lorentz sigma=0, gamma=1, c=1
    # for Gauss sigma=1, gamma=0, c=1


def pseudo_voigt(x, w, n):
    # pseudo-voigt with max=1 and w=FWHM:
    return n * gauss(x, w) + (1-n) * lorentz(x,w)

x = np.arange (-10, 11 , 1e-2)
xfit = np.arange (-12.0, 12.0 , 0.01)

w = 1.0
yL = lorentz(x, w)
yG = gauss(x, w)
yPV = pseudo_voigt(x, w, 0.5)
yV = voigt(x, .562855, .562855)

popt, pcov = curve_fit(voigt, x, yPV, p0=(0.56285008, 0.56291417), sigma=None, bounds=(0,100))
yV = voigt(x, *popt)          
#popt_L, pcov_L = curve_fit(voigt, x, yL, p0=(1., 1e-9, 1.), sigma=None, bounds=(0,100))
#yVL = voigt(xfit, *popt_L)

#popt_G, pcov_G = curve_fit(voigt, x, yG, p0=(1., 2., 1e-9), sigma=None, bounds=(0,100))
#yVG = voigt(xfit, *popt_G)


dL = 0.05
dG = 0.05
wL, wG = np.mgrid[dL:10+dL:dL, dG:10+dG:dG]
yo = voigt(2, wL, wG)
 

#mpl.style.use('default')
plt.close('all')
fig1, ax = plt.subplots(figsize=(14, 8))
ax.grid(b=True, which='both', axis='both')
ax.minorticks_on()
ax.set_title("Gaussian and Lorentzian", fontsize=16)
ax.set_xlabel("x", fontsize=14)
#ax.set_xlim()
ax.set_ylabel("y", fontsize=14)
#ax.set_ylim()
ax.plot(x,yL, '-r', label='Lorentz')
ax.plot(x,yG, '-b', label='Gauss')
ax.plot(x,yPV, '-m', label='Pseudo Voigt')
ax.plot(x,yV, '-g', label='Voigt')
ax.legend()

# levels = MaxNLocator(nbins=15).tick_values(yo.min(), yo.max())
# cmap = plt.get_cmap('PiYG')
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# fig2, ax2 = plt.subplots()

# cf = ax2.contourf(wL[:-1, :-1] + dL/2., wG[:-1, :-1] + dG/2., yo[:-1, :-1], levels=levels, cmap=cmap)
# fig.colorbar(cf, ax=ax2)

plt.show()

#print('Lorentzian:')
#print(popt_L)
#print(pcov_L)
#print('Gaussian:')
#print(popt_G)
#print(pcov_G)
