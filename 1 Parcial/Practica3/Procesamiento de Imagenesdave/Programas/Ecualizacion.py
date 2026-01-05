import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('fig7.jpg',0)
hist,bins = np.histogram(img1.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img1.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histograma'), loc = 'upper right')
plt.show()


cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img1]
 

cv2.imshow('cebras_modificado.png',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()