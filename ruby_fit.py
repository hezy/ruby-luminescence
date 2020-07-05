# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:59:05 2019

@author: Hezy
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.optimize import curve_fit


def lorentz(x, wL):
    # Lorentz with max=1 and w=FWHM:
    gamma = wL
    return 1 / (1 + np.square(x/gamma))


def gauss(x, wG):
    # Gauss with max=1 and w=FWHM
    sigma = wG/np.sqrt(2 * np.log(2))
    return np.exp(- x ** 2 / (2 * sigma ** 2))


def pseudo_voigt(x, x0, w, n, a):
    # pseudo-voigt with max=1 and w=FWHM:
    return a * (n * gauss((x-x0), w) + (1-n) * lorentz((x-x0), w))


def background(x, b0, b1, b2):
    return b0 + b1*x + b2*x**2


def ruby(x, x1, w1, n1, a1, x2, w2, n2, a2, b0, b1, b2):
    return background(x, b0, b1, b2) + \
           pseudo_voigt(x, x1, w1, n1, a1) + \
           pseudo_voigt(x, x2, w2, n2, a2)


def pressure(wl):
    A = 1904.0
    B = 7.665
    wl0 = 693.516
    return A/B * (((wl-wl0)/wl0 + 1)**B - 1)


# Genral figure preperation
plt.close('all')
plt.rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

# read data from csv file
files = glob.glob('./*.txt')
for file in sorted(files):

    data = read_csv(file, skiprows=5, header=None,
                    sep='\t', lineterminator='\n')
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]

    guess = data.loc[data[1].idxmax()]
    h_guess2 = guess[1]
    h_guess1 = h_guess2 * 0.6
    wl_guess2 = guess[0]
    wl_guess1 = wl_guess2-1.58

    popt, pcov = curve_fit(ruby, x, y,
                           p0=(wl_guess1, 0.62, 0.5, h_guess1,
                               wl_guess2, 0.696, 0.5, h_guess2,
                               0, 0, 0),
                           bounds=((650, 0.01, 0.01, 0.01,
                                    650, 0.01, 0.01, 0.01,
                                    -np.inf, -np.inf, -np.inf),
                                   (720, 30, 1, +np.inf,
                                    720, 30, 1, +np.inf,
                                    +np.inf, +np.inf, +np.inf)))
    perr = np.sqrt(np.diag(pcov))
#    print(popt)
#    print(pcov)

    yfit = ruby(x, *popt)
    ydif = yfit - y

    # create figure
    title = 'Pressure measurement by ruby luminescence'
    lamb = '%s' % float('%.6g' % popt[4])
    press = '%s' % float('%.4g' % pressure(popt[4]))
    subtitle = '$ \\lambda $ = ' + lamb + ' nm  ,  P = ' + press + ' GPa'

    plt.figure(figsize=(11.7, 8.3), dpi=144)

    plt.suptitle(title, y=0.98, fontsize=20)
    plt.title(subtitle, y=1, fontsize=20)

    plt.plot(x, y - background(x, *popt[8:11]), '.',
             label='experiment')
    plt.plot(x, yfit - background(x, *popt[8:11]), '-',
             label='fit')
#    plt.plot(x, background(x, *popt[8:11]),
#             label='background')
#    plt.plot(x, background(x, *popt[8:11]) + pseudo_voigt(x, *popt[0:4]),
#             label='peak 1')
#    plt.plot(x, background(x, *popt[8:11]) + pseudo_voigt(x, *popt[4:8]),
#             label='peak 2')
    plt.plot(x, ydif, '-', label='residual')

    # arange figure
    plt.grid(True)
    plt.minorticks_on()
    plt.legend(loc='best')
    plt.xlabel('wavelength (nm)', fontsize=18)
    plt.ylabel('intesity (arb.)', fontsize=18)

    plt.figtext(0.15, 0.15, file, ha='left', va='center')

    plt.savefig(file.strip('.\\').strip('/').strip('.txt'))
    plt.show()
