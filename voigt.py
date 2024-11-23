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


def lorentz(x, gamma):
    # Lorentz with max=1 and w=FWHM=1: return 1. /(1. + np.square(x/w))
    # normolized Lorentz (integral = 1):
    return gamma / np.pi / (np.square(x) + np.square(gamma))


def gauss(x, sigma):
    # Gauss with max=1 and w=FWHM=1: return np.exp(-np.log(2.)*np.square(x/w))
    # normolized Gauss (integral = 1):
    return (np.exp(-np.square(x) / 2 / np.square(sigma))) / (sigma * np.sqrt(2 * np.pi))


def voigt(x, sigma, gamma, c):
    # Voigt with max = 1 and FWHM=1: return c * np.real(wofz(np.sqrt(np.log(2))*(a*x/2 + 1j*b)))
    # normolized Voigt (integral = 1):
    return (
        c
        * np.real(wofz((x + 1j * gamma) / (sigma * np.sqrt(2))))
        / (sigma * np.sqrt(2 * np.pi))
    )
    # for Lorentz sigma=0, gamma=1, c=1
    # for Gauss sigma=1, gamma=0, c=1


x = np.arange(-12.0, 12.0, 0.2)
xfit = np.arange(-12.0, 12.0, 0.01)

yL = lorentz(x, 1.0)
yG = gauss(x, 1.0)
yVtest = voigt(xfit, 1.0, 1.0, 1.0)

popt_L, pcov_L = curve_fit(voigt, x, yL, p0=(1e-9, 1.0, 1.0), sigma=None)
yVL = voigt(xfit, *popt_L)


popt_G, pcov_G = curve_fit(voigt, x, yG, p0=(1.0, 1e-9, 1.0), sigma=None)
yVG = voigt(xfit, *popt_G)

plt.close("all")
fig, ax = plt.subplots(figsize=(14, 8))
ax.grid(b=True, which="both", axis="both")
ax.minorticks_on()
ax.set_title("Gaussian and Lorentzian", fontsize=16)
ax.set_xlabel("x", fontsize=14)
# ax.set_xlim()
ax.set_ylabel("y", fontsize=14)
# ax.set_ylim()
ax.plot(x, yL, "or")
ax.plot(xfit, yVL, "-r")
ax.plot(x, yG, "ob")
ax.plot(xfit, yVG, "-b")

ax.plot(xfit, yVtest, "-g")
plt.show()

print("Lorentzian:")
print(popt_L)
print(pcov_L)
print("Gaussian:")
print(popt_G)
print(pcov_G)
