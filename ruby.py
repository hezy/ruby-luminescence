# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:59:05 2019

@author: Hezy
"""

import numpy as np
from pandas import read_csv
from scipy.special import wofz
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# read data from csv file
data = read_csv('6.6GPa-a.csv', skiprows=None, header=None, sep='\t',
                lineterminator='\n', names=["lambda", "intensity"])
x = data.iloc[:, 0]
y = data.iloc[:, 1]


def lorentz(x, gamma):
    return gamma / np.pi / (np.square(x) + np.square(gamma))


def gauss(x, sigma):
    return (np.exp(-np.square(x)/2/np.square(sigma)))/(sigma*np.sqrt(2*np.pi))


def voigt(x, sigma, gamma):
    return np.real(wofz((x + 1j*gamma)/(sigma * np.sqrt(2)))) /(sigma * np.sqrt(2*np.pi))


def ruby(x, x1, s1, g1, a1, x2, s2, g2, a2, b):
    return b + a1*voigt((x-x1), s1, g1) + a2*voigt((x-x2), s2, g2)


popt, pocv = curve_fit(ruby, x, y, p0=(692.2, 0.178, 0.1, 2689.0,
                                       693.7, 0.247, 0.1, 6058.0,
                                       90.4))
print(popt)
print(pocv)

yfit = ruby(x, *popt)
ydif = yfit - y

# create figure
fig, ax = plt.subplots(figsize=(14, 8))
plt.plot(x, y, '.b', label='experiment')
plt.plot(x, yfit, '-r', label='fit')
plt.plot(x, ydif, '-g', label='difference')

# arange figure
ax.grid(True)
ax.legend(loc='best')
ax.set_title('Ruby')
ax.set_xlabel('wavelength (nm)')
ax.set_ylabel('intesity (arb.)')

plt.show()
