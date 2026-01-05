
import cv2

img1=cv2.imread('Canada.jpg') 
img2=cv2.imread('Manzana.jpg')
img1_1=cv2.resize(img1,(300,300),interpolation= cv2.INTER_CUBIC) 
img2_2=cv2.resize(img2,(300,300),interpolation= cv2.INTER_CUBIC)
#Suma 
suma=cv2.add(img1_1,img2_2)
#Resta
resta=cv2.subtract(img1_1,img2_2)
#
cv2.imshow('imagen 1',img1_1) 
cv2.imshow('IMagen 2',img2_2) 
cv2.imshow('Suma',suma)
cv2.imshow('Resta',resta)                                                 
cv2.waitKey(0)
cv2.destroyAllWindows()