import numpy as np
import cv2 
import math

imagen01 = cv2.imread('tenemos.jpg')
#imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
print(imagen01.shape)
cv2.imshow('Imagen de entrada 01: ',imagen01)

imagen02 = cv2.imread('no te creo.jpg')
#imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
print(imagen02.shape)
cv2.imshow('Imagen de entrada 02: ',imagen02)

alto01, ancho01, ncolores01 = imagen01.shape
alto02, ancho02, ncolores02 = imagen02.shape

x_T = 150
y_T = 50

imagenSuma = np.zeros((int(alto02), int(ancho02), ncolores02),dtype='uint8')
for y in range(alto01):
    for x in range(ancho01):
        if (y + y_T) < alto02 and (x + x_T) < ancho01:
            imagenSuma[y,x,:] = imagen01[y,x,:] - imagen02[y,x,:]
 
cv2.imshow('Imagen de salida', imagenSuma)
cv2.waitKey(0)
cv2.destroyAllWindows()