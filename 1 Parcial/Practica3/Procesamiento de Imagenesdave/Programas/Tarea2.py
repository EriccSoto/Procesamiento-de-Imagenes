import cv2
import numpy as np
import matplotlib.pyplot as plt

img =cv2.imread("Lena.png",0)
inte= []

for i in range(255):
    inte.append(0)
    
for i in range(img.shape[0]):
    
    for j in range(img.shape[1]):
        
        inte[img.item(i,j)]=inte[img.item(i,j)]+1
        
for i in inte:
	print (i)
#print("- promedio (media) de los elementos:")
#print(np.mean(inte))