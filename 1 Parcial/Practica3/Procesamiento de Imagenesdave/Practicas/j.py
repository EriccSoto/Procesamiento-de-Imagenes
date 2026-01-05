import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage.transform as tf
import math
 
lena = cv2.imread('A.jpg',0)
 
f= np.fft.fft2(lena)
fshift = np.fft.fftshift (f)
print (fshift.shape, fshift.dtype)
Espectro = np. log (np.abs(fshift))
 
 
plt.title('Espectro')
plt.imshow(Espectro,cmap ='gray')
 
 
rows,cols =lena.shape
crow,ccol = rows//2, cols//2
mask = np.zeros((rows,cols),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30]=1
print (fshift.shape)
fshift =fshift*mask
f_ishift = np.fft.ifftshift(fshift)
lena_filtro = np.fft.ifft2(f_ishift)
lena_filtro = np.abs(lena_filtro)
 
plt.title('Lena LPF')
plt.imshow(lena_filtro,cmap='gray')