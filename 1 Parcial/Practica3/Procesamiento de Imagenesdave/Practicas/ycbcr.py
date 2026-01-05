import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt
ima = mpimg.imread('Lena.png')
ima_r = ima[:, :, 0]
ima_g = ima[:, :, 1]
ima_b = ima[:, :, 2]

#Obtener brillo, que es una copia en escala de grises de la imagen original
ima_y = 0.256789 * ima_r + 0.504129 * ima_g + 0.097906 * ima_b + 16
 #Obtener el componente azul
ima_cb = -0.148223 * ima_r - 0.290992 * ima_g + 0.439215 * ima_b + 128
 #Obtener el componente rojo
ima_cr = 0.439215 * ima_r - 0.367789 * ima_g - 0.071426 * ima_b + 128
 
# Combina los tres componentes juntos
ima_rgb2ycbcr = np.zeros(ima.shape)
ima_rgb2ycbcr[:,:,0] = ima_y
ima_rgb2ycbcr[:,:,1] = ima_cb
ima_rgb2ycbcr[:,:,2] = ima_cr

plt.imshow(ima_y)
plt.title("Y")
plt.show()
plt.imshow(ima_cb)
plt.title("CB")
plt.show()
plt.imshow(ima_cr)
plt.title("CR") 
plt.show()

ima_ycbcr2rgb = np.zeros(ima.shape)
ima_ycbcr2rgb[:,:,0] = 1.164383 * (ima_y-16) + 1.596027 * (ima_cr-128)
ima_ycbcr2rgb[:,:,1] = 1.164383 * (ima_y-16) - 0.391762 * (ima_cb-128)- 0.812969 * (ima_cr-128)
ima_ycbcr2rgb[:,:,2] = 1.164383 * (ima_y-16) + 2.017230 * (ima_cb-128)

plt.imshow(ima_ycbcr2rgb[:,:,0])
plt.title("R")
plt.show()
plt.imshow(ima_ycbcr2rgb[:,:,1])
plt.title("V")
plt.show()
plt.imshow(ima_ycbcr2rgb[:,:,2])
plt.title("A")
plt.show()