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

angulo = math.pi / 4
xre = 250
yre = 250
imagenRotacionpv = np.zeros((int(alto), int(ancho), ncolores),dtype='uint8')
for y in range(alto):
  for x in range(ancho):
    xr = abs(int(xre + (x-xre) * math.cos(angulo) - (y-yre) * math.sin(angulo)))
    yr = abs(int(yre + (y-yre) * math.cos(angulo) + (x-xre) * math.sin(angulo)))
    # print(xr, yr)
    if xr > 0 and yr > 0 and xr < ancho and yr < alto:
      imagenRotacionpv[yr, xr, :] = imagen[y, x, :]
cv2.imshow('Imagen de salida', imagenRotacionpv)
cv2.waitKey(0)
cv2.destroyAllWindows()

