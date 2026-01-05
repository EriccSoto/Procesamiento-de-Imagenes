
import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('Lena.png')
b, g ,r =cv2.split(img)

zeros = np.zeros(img.shape[:2], dtype = "uint8")          
merged_r = cv2.merge ([zeros, zeros, r]) #El componente del canal es cero se puede entender como una matriz cero
merged_g = cv2.merge ([zeros, g, zeros])
merged_b = cv2.merge ([b, zeros, zeros])
cv2.imshow('image',img)
cv2.imshow("Rojo",merged_r)
cv2.imshow("Verde",merged_g)
cv2.imshow("Azul",merged_b)
cv2.waitKey(0)