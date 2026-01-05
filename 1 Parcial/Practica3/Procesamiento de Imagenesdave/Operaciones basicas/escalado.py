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

Sx = 0.5
Sy = 0.5
imagenEscalado = np.zeros((int(alto * Sy), int(ancho * Sx), ncolores),dtype='uint8')
for y in range(alto):
  for x in range(ancho):
      imagenEscalado[int(y * Sy), int(x * Sx), :] = imagen[y, x, :]
cv2.imshow('Imagen de salida',imagenEscalado)
cv2.waitKey(0)
cv2.destroyAllWindows()

