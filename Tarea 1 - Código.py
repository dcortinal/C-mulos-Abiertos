# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:05:29 2024

@author: damia
"""
import matplotlib.pyplot as plt
import numpy as np 

Names =['Alpha Centauri A', '20 Tauri','WR 22']
Temps = np.array([5260, 12300, 44700])
h = 6.626e-34
c = 299792458
Kb = 1.3806e-23
Spectra = np.array([[0.3e-6, 0.4e-6], [0.35e-6, 0.55e-6], [0.45e-6, 0.7e-6]])
Spectral_Magnitudes = np.zeros((len(Temps), len(Spectra)))
colors = ['darkviolet','deepskyblue','slategrey']
index = ['U','B','V']

def planck(T, λ):
    return (8*np.pi*h*c*(λ**(-5)))*(1/(np.exp((h * c) / (λ * Kb * T)) - 1))

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(24, 14))

for i in range(len(Temps)):
    for j in range(len(Spectra)):
        x_total = np.linspace(0,1e-6,1000)
        x_values = np.linspace(Spectra[j, 0], Spectra[j, 1], 1000)
        Specific_Intensities = planck(Temps[i], x_values)
        Spectral_Intensities = np.trapz(Specific_Intensities, x_values)
        Spectral_Magnitudes[i, j] = -2.5*np.log10(Spectral_Intensities)
        label = [f'$m_{index[j]}$ = {Spectral_Magnitudes[i, j]:.2f}']
        axs[i].fill_between(x_total, 0, planck(Temps[i], x_total), where=(x_total >= Spectra[j, 0]) & (x_total <= Spectra[j, 1]), color=colors[j], alpha=0.4, label=label)
     
    axs[i].set_title(Names[i],fontsize=24)
    axs[i].grid(True)
    axs[i].plot(x_total, planck(Temps[i], x_total),c='k',label=f'T = {Temps[i]}K')
    axs[i].legend(fontsize=22,loc='upper right')
axs[0].set_ylabel('$B(\\lambda,T) [erg \cdot s^{-1} \cdot sr^{-1} \cdot cm^{-2}  \cdot Hz^{-1}]$',fontsize = 22)
axs[1].set_xlabel('$\\lambda [m]$',fontsize = 22)
fig.suptitle('Magnitudes Específicas Según la Intensidad de la Estrella', fontsize=32)

U_B_mag = Spectral_Magnitudes[:,0]-Spectral_Magnitudes[:,1]
B_V_mag = Spectral_Magnitudes[:,1]-Spectral_Magnitudes[:,2]

U_B_real = np.array([0.24,-0.4,-0.82])
B_V_real = np.array([0.71,-0.07,0.08])
