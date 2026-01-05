import cv2
import numpy as np
import matplotlib.pyplot as plt


def negativo2( img ):
    neg = 255-img
    return neg


def negativo3( img ):
    neg = np.zeros( img.shape , dtype=np.uint8)
    for f in range( img.shape[0] ):
        for c in range( img.shape[1] ) :
            neg[f,c] = 255 - img[f,c];
    return neg


imagen = cv2.imread('Paisaje.jpg')     #BGR
neg = cv2.bitwise_not(imagen)
neg2 = negativo2(imagen)
neg3 = negativo3(imagen)


cv2.imshow("Imagen - Negativo", np.hstack((imagen, neg)))
cv2.imshow("Imagen - Negativo", np.hstack((neg2, neg3)))
cv2.waitKey();
cv2.destroyAllWindows()



print("\nImg: ", imagen[0])
print("\nNeg: ", neg[0])
print("\nNeg2: ", neg2[0])
print("\nNeg3: ", neg3[0])
print(type( imagen ))
print(type( neg ))
print(type( neg2 ))
print(type( neg3 ))

