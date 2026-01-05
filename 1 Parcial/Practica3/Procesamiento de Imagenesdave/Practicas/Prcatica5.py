import cv2
import numpy as np
 
img = cv2.imread('C.bmp',0)
kernel = np.ones((7,7),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
dilatacion = cv2.dilate(img,kernel,iterations = 1)
apertura = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cierre = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


cv2.imshow("Erosion",erosion)
cv2.imshow("Dilatacion",dilatacion)
cv2.imshow("Apertura",apertura)
cv2.imshow("Cierre",cierre)
cv2.imshow("Esqueleto",gradient)

cv2.waitKey(0)