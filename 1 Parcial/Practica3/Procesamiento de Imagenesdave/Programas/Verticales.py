import cv2
import numpy as np
import matplotlib.pyplot as plot
import random

img = cv2.imread("lena.png",0)
cv2.imshow("Imagen Original ", img)
cv2.waitKey(0)

alto, ancho = img.shape

img2 = np.zeros((alto,ancho,3),np.uint8)

for i in range(0,alto):
    for j in range(0,ancho):
        img2[i,j,:] = (img.item(i, j))
i=8
while i <= ancho:
    tonB = random.randrange(255)
    tonG = random.randrange(255)
    tonR = random.randrange(255)

    for j in range(0,i):
        if( (j+(i*3)-16)>=ancho ):
            break
        else: 
            img2[:,j+(i*3)-16,0] = (tonR)
            img2[:,j+(i*3)-16,1] = (tonG)
            img2[:,j+(i*3)-16,2] = (tonB)
    i *= 2

cv2.imshow("Verticales", img2)
cv2.waitKey(0)

