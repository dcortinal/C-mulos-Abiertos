# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:05:29 2024

@author: damia
"""
import matplotlib.pyplot as plt
import numpy as np 

Names =['$\\alpha$ Centauri A', '$\\sigma$ Orionis E','$\\theta^1$ Orionis C1']
temps = np.array([5260, 25000, 39000])
M_sun = [5.61,5.44,4.81]
F_sun = [94.323,333.548,445.642]
d = [1.35,352,460]
h = 6.626e-34
c = 299792458
Kb = 1.3806e-23
spectra = np.array([[0.3e-6, 0.4e-6], [0.35e-6, 0.55e-6], [0.45e-6, 0.7e-6]])
λ = np.linspace(spectra[0,0],spectra[2,1],1000)
Spectral_Magnitudes = np.zeros((len(temps), len(spectra)))
Luminosities = np.zeros(len(temps))
colors = ['darkviolet','deepskyblue','slategrey']
index = ['U','B','V']

def planck(T, λ):
    return (2*h*(λ**(-5))*c)*(1/(np.exp((h * c) / (λ*Kb * T)) - 1))

def flux_density(T,λ,filter_):
    return np.pi*np.trapz(planck(T, λ),filter_)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(24, 14))
x = np.linspace(0,1e-6,1000)

for i in range(len(temps)):
    for j in range(len(spectra)):
        F = flux_density(temps[i],λ,spectra[j])
        M = -2.5*np.log10(F/F_sun[j])
        m = 5*np.log10(d[i])-5+M
        Spectral_Magnitudes[i,j] = m
        label = [f'${index[j]}$ = {Spectral_Magnitudes[i, j]:.2f}']
        axs[i].fill_between(x, 0, planck(temps[i], x), where=(x >= spectra[j, 0]) & (x <= spectra[j, 1]), color=colors[j], alpha=0.4, label=label)
     
    axs[i].set_title(Names[i],fontsize=24)
    axs[i].grid(True)
    axs[i].plot(x, planck(temps[i], x),c='k',label=f'T = {temps[i]}K')
    axs[i].legend(fontsize=22,loc='upper right')
axs[0].set_ylabel('$B(\\lambda,T) [Wm^{-3}]$',fontsize = 26)
axs[1].set_xlabel('$\\lambda [m]$',fontsize = 26)
fig.suptitle('Magnitudes aparentes en la Tierra filtradas en UBV', fontsize=36)

Spectral_Theorical_Magnitudes = np.array([[0.96,0.72,0.01],[5.66,6.38,6.46],[4.20,5.15,5.13]])

UB_exp = Spectral_Magnitudes[:,1] - Spectral_Magnitudes[:,0]
BV_exp = Spectral_Magnitudes[:,2] - Spectral_Magnitudes[:,1]
UB = Spectral_Theorical_Magnitudes[:,1] - Spectral_Theorical_Magnitudes[:,0]
BV = Spectral_Theorical_Magnitudes[:,2] - Spectral_Theorical_Magnitudes[:,1]