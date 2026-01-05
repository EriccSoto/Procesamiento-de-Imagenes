import cv2
import numpy as np
import matplotlib.pyplot as plt

 
img = cv2.imread('ima1.jpg')
print (img)
img_shape = img.shape
y = img_shape[0]
x = img_shape[1]
 
for fila in range(x):
    for columna in range(y):

    	break

cv2.imshow('Imagen', img)
print(img[columna][fila])
print("Promedio:");
Promedio=np.mean(img)
print(Promedio)
print("Desviacion Estandar:");
Desviacion=np.std(img)
print(Desviacion)
cv2.waitKey(0)
cv2.destroyAllWindows()