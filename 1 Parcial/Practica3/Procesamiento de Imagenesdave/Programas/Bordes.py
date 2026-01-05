import numpy as np
import cv2

im = cv2.imread("Canada.jpg")
cv2.imshow("Imagen Original ", im)
row, col = im.shape[:2]
bottom = im[row-2:row, 0:col]
mean = cv2.mean(bottom)[0]

bordersize = 5
border = cv2.copyMakeBorder(
    im,
    top=bordersize,
    bottom=bordersize,
    left=bordersize,
    right=bordersize,
    borderType=cv2.BORDER_CONSTANT,
    value=[mean, mean, mean]
)

cv2.imshow('Borde ', border)
cv2.waitKey()
cv2.destroyAllWindows()

