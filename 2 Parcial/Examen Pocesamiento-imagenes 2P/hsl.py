import cv2
import numpy as np
import math

def rgb_a_hsi(imagen_path):
    imagen = cv2.imread(imagen_path)
    alto, ancho, canales = imagen.shape
    canal_h = np.zeros((alto, ancho, 3), dtype=np.uint8)
    canal_s = np.zeros((alto, ancho, 3), dtype=np.uint8)
    canal_i = np.zeros((alto, ancho, 3), dtype=np.uint8)
    imagen_fusionada = np.zeros((alto, ancho, 3), dtype=np.uint8)

    for i in range(alto):
        for j in range(ancho):
            valor_r = imagen[i, j][2] / 255.0
            valor_g = imagen[i, j][1] / 255.0
            valor_b = imagen[i, j][0] / 255.0

            num = 0.5 * ((valor_r - valor_g) + (valor_r - valor_b))
            den = math.sqrt((valor_r - valor_g)**2 + (valor_r - valor_b)*(valor_g - valor_b)) + 1e-8
            theta = math.acos(num / den)

            if valor_b <= valor_g:
                h = theta
            else:
                h = 2 * math.pi - theta

            h = h / (2 * math.pi)

            min_rgb = min(valor_r, valor_g, valor_b)
            sum_rgb = valor_r + valor_g + valor_b
            if sum_rgb == 0:
                s = 0
            else:
                s = 1 - (3 * min_rgb / sum_rgb)

            i_valor = sum_rgb / 3

            canal_h[i, j] = [int(h * 255), int(h * 255), int(h * 255)]
            canal_s[i, j] = [int(s * 255), int(s * 255), int(s * 255)]
            canal_i[i, j] = [int(i_valor * 255), int(i_valor * 255), int(i_valor * 255)]

            r_fusionado = i_valor * (1 + s * math.cos(h * 2 * math.pi) / math.cos((1/3 * math.pi) - h * 2 * math.pi))
            g_fusionado = i_valor * (1 + s * (1 - math.cos(h * 2 * math.pi) / math.cos((1/3 * math.pi) - h * 2 * math.pi)))
            b_fusionado = 3 * i_valor - (r_fusionado + g_fusionado)

            r_fusionado = max(0, min(1, r_fusionado))
            g_fusionado = max(0, min(1, g_fusionado))
            b_fusionado = max(0, min(1, b_fusionado))

            imagen_fusionada[i, j] = [int(b_fusionado * 255), int(g_fusionado * 255), int(r_fusionado * 255)]

    cv2.imshow('Imagen original', imagen)
    #cv2.imshow('Canal H', canal_h)
    #cv2.imshow('Canal S', canal_s)
    cv2.imshow('Canal I', canal_i)
    #cv2.imshow('Imagen fusionada HSI', imagen_fusionada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

rgb_a_hsi('Baboon.PNG')
