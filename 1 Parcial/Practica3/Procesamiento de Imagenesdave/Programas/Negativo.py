import numpy as np
import cv2

img = cv2.imread('Manzana.jpg',0)    
filas, columnas = img.shape                                                                          
negativo = np.zeros((filas, columnas), dtype=np.uint8)             
  
                                                                                          
for a in range(0, filas):  
		for b in range(0, columnas):                                                                                 
                 negativo[a,b] = 255 - img[a,b] 

              
cv2.imshow('ORIGINAL', img)       
cv2.imshow('NEGATIVO', negativo)       
cv2.waitKey(0)
cv2.destroyAllWindows()

