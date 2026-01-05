import cv2

image = cv2.imread('tenemos.jpg')

#recortar
imageOut = image[60:220,200:480]
cv2.imshow('Imagen de entrada',image)
cv2.imshow('Imagen de salida',imageOut)
cv2.waitKey(0)
cv2.destroyAllWindows()