
import cv2

src = cv2.imread("Ejemplo_1.jpg")
imgGray = cv2.imread("Ejemplo_1.jpg",0)

cv2.imshow("src", src)
cv2.imshow("result", imgGray)
#Esperando mostrar
cv2.waitKey(0)