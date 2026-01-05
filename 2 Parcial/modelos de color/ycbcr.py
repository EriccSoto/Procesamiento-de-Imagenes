import cv2
import numpy as np

def rgb_a_ycbcr(imagen_path):
    imagen = cv2.imread(imagen_path)
    alto, ancho, canales = imagen.shape
    canal_y = np.zeros((alto, ancho, 3), dtype=np.uint8)
    canal_cb = np.zeros((alto, ancho, 3), dtype=np.uint8)
    canal_cr = np.zeros((alto, ancho, 3), dtype=np.uint8)
    imagen_fusionada = np.zeros((alto, ancho, 3), dtype=np.uint8)

    for i in range(alto):
        for j in range(ancho):
            valor_r = imagen[i, j][2]
            valor_g = imagen[i, j][1]
            valor_b = imagen[i, j][0]
            r = valor_r / 255.0
            g = valor_g / 255.0
            b = valor_b / 255.0
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cb = (b - y) * 0.564 + 0.5
            cr = (r - y) * 0.713 + 0.5
            
            canal_y[i, j] = [int(y * 255), int(y * 255), int(y * 255)]
            canal_cb[i, j] = [255,int(cb * 255), int(cb * 255)] 
            canal_cr[i, j] = [int(cr * 255), int(cr * 255),255]  

            r_fusionado = int((y + 1.403 * (cr - 0.5)) * 255)
            g_fusionado = int((y - 0.344 * (cb - 0.5) - 0.714 * (cr - 0.5)) * 255)
            b_fusionado = int((y + 1.773 * (cb - 0.5)) * 255)
            r_fusionado = max(0, min(255, r_fusionado))
            g_fusionado = max(0, min(255, g_fusionado))
            b_fusionado = max(0, min(255, b_fusionado))
            imagen_fusionada[i, j] = [b_fusionado, g_fusionado, r_fusionado]

    cv2.imshow('Imagen original', imagen)
    cv2.imshow('Canal Y', canal_y)
    cv2.imshow('Canal Cb', canal_cb)
    cv2.imshow('Canal Cr', canal_cr)
    cv2.imshow('Imagen fusionada YCbCr', imagen_fusionada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

rgb_a_ycbcr('skz.jpg')
