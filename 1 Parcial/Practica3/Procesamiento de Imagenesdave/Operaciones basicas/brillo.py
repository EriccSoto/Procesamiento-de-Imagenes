import numpy as np
import cv2 
import math

imagen = cv2.imread('tenemos.jpg')
#imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
print(imagen.shape)
cv2.imshow('Imagen de entrada: ',imagen)

alto, ancho, ncolores = imagen.shape
print(ancho)
print(alto)
print(ncolores)
      
imagenBrillo = np.zeros((int(alto), int(ancho), ncolores),dtype='uint8')
for y in range(alto):
    for x in range(ancho):
      imagenBrillo[y, x, :] = imagen[y, x, :] + 30 
 
cv2.imshow('Imagen de salida', imagenBrillo)
cv2.waitKey(0)
cv2.destroyAllWindows()