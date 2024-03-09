# -*- coding: utf-8 -*-
import numpy as np
from astropy.io import fits
from glob import glob
import matplotlib.pyplot as plt
from astropy.stats import biweight_location, biweight_scale
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.ndimage import shift
from scipy.optimize import minimize
from matplotlib.patches import Circle

glob("*")

files = glob(r'C:\Users\damia\OneDrive\Documentos\1. Damián\Universidad de los Andes\Cúmulos Abiertos\Laboratorios\1\Lab 1 - Datos/*.fits')

np.set_printoptions(linewidth=120)

imagesd = [fits.getdata(f) for f in files]
images = []
images.append(imagesd[1])
images.append(imagesd[0])

plt.rcParams["image.cmap"] = "viridis" 
plt.rcParams["image.origin"] = "lower"

flattened = images[1].ravel()
flattened.shape

def my_imshow(img,flat_data,n,**kwargs):
    mad = biweight_scale(flat_data)
    med = biweight_location(flat_data)
    img_vmin = med - n*mad
    img_vmax = med + n*mad
    return plt.imshow(img,vmin=img_vmin,vmax=img_vmax,**kwargs),med,mad,img_vmax

fig, ax = plt.subplots(ncols=2, figsize=(10,6))
for i in range(2):
    plt.sca(ax[i])
    my_imshow(images[0],flattened,10)[0]
    plt.colorbar()
    if i == 0:
        plt.xlabel("UT = 3:30:24",fontsize = 20)
    else: 
        plt.xlabel("UT = 3:33:18",fontsize = 20)
    
    # Agregar un círculo rojo en la región específica
    circle = Circle((140, 260), 10, color='red', fill=False)  # Centro en (140, 260), radio de 10
    ax[i].add_patch(circle)

plt.tight_layout()
fig.suptitle("Imágenes tomadas y procesadas", fontsize=25, x=0.5, y=1.05, ha='center')

fig,ax = plt.subplots(ncols=2,figsize=(10,10))
for i in range(2):
   plt.sca(ax[i])
   my_imshow(images[i][250:270,130:150],images[i][250:270,130:150].ravel(),10)[0]
   plt.axvline(12.5,c='r')
   plt.axhline(9.3,c='r')
   if i == 0:
       plt.xlabel("UT = 3:30:24",fontsize = 20)
   else: 
       plt.xlabel("UT = 3:33:18",fontsize = 20)
plt.tight_layout()
fig.suptitle("Corrimiento del centro en la estrella seleccionada", fontsize=25, x=0.5, y=0.8, ha='center')  

sum_y = [image[250:270,130:150].sum(axis=0) for image in images]
sum_x = [image[250:270,130:150].sum(axis=1) for image in images]

new_image_0 = images[0][250:270,130:150]
new_image_1 = images[1][250:270,130:150]

def voigt(x, B, x0, s1, s2, A):
    return A + B*voigt_profile(x-x0,s1,s2)

y_fits_V,x_fits_V = [],[]

for i in range(2):
  x = np.linspace(130,150,len(sum_x[i]))
  max_x = x[np.argmax(sum_x[i])]
  initial_guess_V1 = [10000,max_x,1,1,1]
  params_V1, covariance_V1 = curve_fit(voigt,x, sum_x[i], p0=initial_guess_V1)
  B_fit_V1, x0_fit_V1, s1_fit_V1, s2_fit_V1, A_fit_V1 = params_V1
  x_fit_V1 = np.linspace(min(x), max(x), 1000)
  y_fit_V1 = voigt(x_fit_V1, B_fit_V1, x0_fit_V1, s1_fit_V1, s2_fit_V1, A_fit_V1)
  y_fits_V.append(y_fit_V1)

for i in range(2):
  y = np.linspace(250,270,len(sum_y[i]))
  max_y = y[np.argmax(sum_y[i])]
  initial_guess_V2 = [20000,max_y,1,1,1]
  params_V2, covariance_V2 = curve_fit(voigt,y, sum_y[i], p0=initial_guess_V2)
  B_fit_V2, x0_fit_V2, s1_fit_V2, s2_fit_V2, A_fit_V2 = params_V2
  x_fit_V2 = np.linspace(min(y), max(y), 1000)
  y_fit_V2 = voigt(x_fit_V2, B_fit_V2, x0_fit_V2, s1_fit_V2, s2_fit_V2, A_fit_V2)
  x_fits_V.append(y_fit_V2)

fig,ax = plt.subplots(ncols=2,figsize=(14,7),sharey=True)
fig.suptitle("Perfil de Voigt de la estrella seleccionada", fontsize=28, x=0.5, y=0.95, ha='center')
plt.sca(ax[0])
plt.plot(x_fit_V2,x_fits_V[0],c='r',zorder = 0)
plt.plot(x_fit_V2,x_fits_V[1],c='b',zorder = 0)
plt.scatter(y,sum_y[0],c='crimson',zorder = 1,label = 'UT = 3:30:24')
plt.scatter(y,sum_y[1],c='navy',zorder = 1,label = 'UT = 3:33:18')
plt.xlabel("Píxel (x)",fontsize=25)
plt.ylabel("Intensidad",fontsize=25)
plt.grid(True)
plt.legend(fontsize=18,loc='upper left')

plt.sca(ax[1])
plt.plot(x_fit_V1,y_fits_V[0],c='r',zorder = 0)
plt.plot(x_fit_V1,y_fits_V[1],c='b',zorder = 0)
plt.scatter(x,sum_x[0],c='crimson',zorder = 1,label = 'UT = 3:33:18')
plt.scatter(x,sum_x[1],c='navy',zorder = 1,label = 'UT = 3:30:24')
plt.xlabel("Píxel (y)",fontsize=25)
plt.grid(True)
plt.legend(fontsize=18,loc='upper left')

unshifted_image = new_image_1
initial_guess = [0,0]

def error(shifts):
    shifted_image = shift(unshifted_image, shifts)
    aligned_diff = new_image_0 - shifted_image
    return np.sum(aligned_diff**2)

result = minimize(error, initial_guess, method='Powell')
optimal_shift = result.x
shifted_image = shift(new_image_1,optimal_shift)
shifted_complete = shift(images[1],optimal_shift)
fig,ax = plt.subplots(ncols=2,nrows=2,figsize=(10,10))
fig.suptitle("Comparación anterior y posterior al alineamiento", fontsize=25, x=0.5, y=0.95, ha='center')  
for i in range(2):
  for j in range(2):
    plt.sca(ax[i,j])
    if i == 0:
      
      if j == 0:
          plt.ylabel("Estrella seleccionada",fontsize=20)
          plt.title("Anterior",fontsize=20)
          p = my_imshow(new_image_0 - new_image_1,(new_image_0 - new_image_1).ravel(),10)[0]
          plt.colorbar(p)
      else:
          plt.title("Posterior",fontsize=20)
          p = my_imshow(new_image_0 - shifted_image,(new_image_0- shifted_image).ravel(),10)[0]
          plt.colorbar(p)
    else:
      
      if j == 0:
          plt.ylabel("Imagen inicial",fontsize=20)
          p = my_imshow(images[0]  - images[1],(images[0] - images[1]).ravel(),10)[0]
          plt.colorbar(p)
      else:
          p = my_imshow(images[0] * 1.08224535 - shifted_complete,(images[0] - shifted_complete).ravel(),10)[0]
          plt.colorbar(p)
          
print(optimal_shift)
