import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab

ruta=plt.imread("ima1.jpg")
plt.imshow(ruta)
pylab.show()
a=np.array([[-1.0,0.0,1.0],[-1.0,0.0,1.0],[-1.0,0.0,1.0]])
dst=cv2.filter2D(ruta,-1,a)
print(dst.min(),dst.max())
plt.imshow(dst)
plt.imsave("res.jpg",dst)
pylab.show()