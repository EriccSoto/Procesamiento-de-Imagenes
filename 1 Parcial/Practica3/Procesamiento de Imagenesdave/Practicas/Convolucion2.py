import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np
 
img = plt.imread ("ima1.jpg") #Lea la imagen aquí
 
plt.imshow (img) # Muestra la imagen leída
pylab.show()
 
fil = np.array ([[1, 0, -1], # Este es el filtro establecido, que es el núcleo de convolución
                [ 2, 0, -2],
                [  1, 0, -1]])
 
res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
 
plt.imshow (res) #Muestra la imagen después de la convolución
plt.imsave("res.jpg",res) 
pylab.show()