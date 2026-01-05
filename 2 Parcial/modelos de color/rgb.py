import cv2
import numpy as np

def mostrar_canales(imagen):
    
    alto, ancho, canales = imagen.shape
    canal_rojo = np.zeros((alto, ancho, 3), dtype=np.uint8)
    canal_verde = np.zeros((alto, ancho, 3), dtype=np.uint8)
    canal_azul = np.zeros((alto, ancho, 3), dtype=np.uint8)

    for i in range(alto):
        for j in range(ancho):
            valor_b = imagen[i, j][0]
            valor_g = imagen[i, j][1]
            valor_r = imagen[i, j][2]

            canal_rojo[i, j] = [valor_r, 0, 0]  # Canal rojo en RGB
            canal_verde[i, j] = [0, valor_g, 0]  # Canal verde en RGB
            canal_azul[i, j] = [0, 0, valor_b]  # Canal azul en RGB

    cv2.imshow('Imagen original', imagen)
    cv2.imshow("Canal Rojo", canal_rojo)
    cv2.imshow("Canal Verde", canal_verde)
    cv2.imshow("Canal Azul", canal_azul)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


imagen = cv2.imread('skz.jpg')
mostrar_canales(imagen)
