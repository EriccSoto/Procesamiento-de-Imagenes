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

angulo =  math.pi / 4
imagenRotacion = np.zeros((int(alto), int(ancho), ncolores),dtype='uint8')
for y in range(alto):
  for x in range(ancho):
    xr = abs(int(x * math.cos(angulo) - y * math.sin(angulo)))
    yr = abs(int(y * math.cos(angulo) + x * math.sin(angulo)))
    #print(xr, yr)
    if xr > 0 and yr > 0 and xr < ancho and yr < alto:
      imagenRotacion[yr, xr, :] = imagen[y, x, :]
cv2.imshow('Imagen de salida',imagenRotacion)
cv2.waitKey(0)
cv2.destroyAllWindows()

