import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

imagen = cv.imread('Paisaje.jpg')
if imagen is None:
    print('No se puede abrir la imagen: ')
    exit(0)
ima = np.zeros(imagen.shape, imagen.dtype)
contraste = float (input ( '* Ingrese el valor del Contraste en el intervalo 1.0 al 3.0' ))
brillo = int (input ( '* Ingrese el valor del Brillo en el intervalo -100 al 100' )) 
for y in range(imagen.shape[0]):
    for x in range(imagen.shape[1]):
        for c in range(imagen.shape[2]):
            ima[y,x,c] = np.clip(contraste*imagen[y,x,c] + brillo, 0, 255)
cv.imshow('IMagen Original', imagen)
cv.imshow('Imagen Arreglada', ima)
cv.waitKey()
