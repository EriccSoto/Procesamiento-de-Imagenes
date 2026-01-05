import cv2
import numpy as np

def rgb_a_cmyk(imagen_path):
    imagen = cv2.imread(imagen_path)
    alto, ancho, canales = imagen.shape
    canal_cian = np.zeros((alto, ancho, 3), dtype=np.uint8)
    canal_magenta = np.zeros((alto, ancho, 3), dtype=np.uint8)
    canal_amarillo = np.zeros((alto, ancho, 3), dtype=np.uint8)
    canal_negro = np.zeros((alto, ancho, 3), dtype=np.uint8)
    imagen_fusionada = np.zeros((alto, ancho, 3), dtype=np.uint8)

    for i in range(alto):
        for j in range(ancho):
            valor_r = imagen[i, j][2]
            valor_g = imagen[i, j][1]
            valor_b = imagen[i, j][0]
            r = valor_r / 255.0
            g = valor_g / 255.0
            b = valor_b / 255.0
            c = 1 - r
            m = 1 - g
            y = 1 - b
            k = min(c, m, y)
            if k < 1:
                c = (c - k) / (1 - k)
                m = (m - k) / (1 - k)
                y = (y - k) / (1 - k)
            else:
                c = m = y = 0
            canal_cian[i, j] = [int(c * 255), int(c * 255),0]
            canal_magenta[i, j] = [int(m * 255), 0, int(m * 255)]
            canal_amarillo[i, j] = [0, int(y * 255), int(y * 255)]
            canal_negro[i, j] = [int(k * 255), int(k * 255), int(k * 255)]
            c1 = c * (1 - k) + k
            m1 = m * (1 - k) + k
            y1 = y * (1 - k) + k
            r_fusionado = int((1 - c1) * 255)
            g_fusionado = int((1 - m1) * 255)
            b_fusionado = int((1 - y1) * 255)
            r_fusionado = max(0, min(255, r_fusionado))
            g_fusionado = max(0, min(255, g_fusionado))
            b_fusionado = max(0, min(255, b_fusionado))
            imagen_fusionada[i, j] = [b_fusionado, g_fusionado, r_fusionado]

    cv2.imshow('Imagen original', imagen)
    cv2.imshow("Canal Cian", canal_cian)
    cv2.imshow("Canal Magenta", canal_magenta)
    cv2.imshow("Canal Amarillo", canal_amarillo)
    cv2.imshow("Canal Negro", canal_negro)
    cv2.imshow('Imagen fusionada CMYK', imagen_fusionada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

rgb_a_cmyk('skz.jpg')
