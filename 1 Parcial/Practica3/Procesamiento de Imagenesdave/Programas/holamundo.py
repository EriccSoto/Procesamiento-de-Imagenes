import numpy as np
import cv2

img = cv2.imread('Manzana.jpg',0)      #Abrimos la imagen
filas, columnas = img.shape                                                                          #Tamaño de la imagen
negativo = np.zeros((filas, columnas), dtype=np.uint8)             #Creamos una matriz usando una variable  
  
                                                                                          # para una nueva imagen 

for a in range(0, filas):  
		for b in range(0, columnas):                                                 #Con un ciclo recorremos las filas                            for b in range(0, col):                                                 y columnas
                 negativo[a,b] = 255 - img[a,b]         #Con esta operación invertimos el valor de los pixeles

cv2.imshow('ORIGINAL', img)                                #Mostramos la imagen original
cv2.imshow('NEGATIVO', negativo)                       #Mostramos la imagen con los colores negativos
k = cv2.waitKey(0)
cv2.destroyAllWindows()