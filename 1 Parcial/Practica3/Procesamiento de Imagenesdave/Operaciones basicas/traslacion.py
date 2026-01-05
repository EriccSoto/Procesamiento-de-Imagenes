import numpy as np
import cv2 

imagen = cv2.imread('tenemos.jpg')
#imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
print(imagen.shape)
cv2.imshow('Imagen de entrada: ',imagen)

alto, ancho, ncolores = imagen.shape
print(ancho)
print(alto)
print(ncolores)

x_T = 150
y_T = 50
imagenTraslacion = np.zeros((alto + y_T, ancho + x_T, ncolores),dtype='uint8')
for y in range(alto):
  for x in range(ancho):
    #if (y + y_T) < alto and (x + x_T) < ancho:
    imagenTraslacion[y + y_T, x + x_T, :] = imagen[y, x, :]

cv2.imshow('Imagen de salida',imagenTraslacion)
cv2.waitKey(0)
cv2.destroyAllWindows()

