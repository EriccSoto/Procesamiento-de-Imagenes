import cv2
import numpy as np

def rgb_a_lab(imagen_path):
    imagen = cv2.imread(imagen_path)
    alto, ancho, canales = imagen.shape
    canal_l = np.zeros((alto, ancho, 3), dtype=np.uint8)
    canal_a = np.zeros((alto, ancho, 3), dtype=np.uint8)
    canal_b = np.zeros((alto, ancho, 3), dtype=np.uint8)

    def f(t):
        if t > 0.008856:
            return t ** (1/3)
        else:
            return (7.787 * t) + (16 / 116)

    for i in range(alto):
        for j in range(ancho):
            valor_r = imagen[i, j][2] / 255.0
            valor_g = imagen[i, j][1] / 255.0
            valor_b = imagen[i, j][0] / 255.0

            # Corrección gamma inversa
            if valor_r > 0.04045:
                valor_r = ((valor_r + 0.055) / 1.055) ** 2.4
            else:
                valor_r = valor_r / 12.92
            if valor_g > 0.04045:
                valor_g = ((valor_g + 0.055) / 1.055) ** 2.4
            else:
                valor_g = valor_g / 12.92
            if valor_b > 0.04045:
                valor_b = ((valor_b + 0.055) / 1.055) ** 2.4
            else:
                valor_b = valor_b / 12.92

            valor_r *= 100
            valor_g *= 100
            valor_b *= 100

            # RGB a XYZ
            x = valor_r * 0.4124 + valor_g * 0.3576 + valor_b * 0.1805
            y = valor_r * 0.2126 + valor_g * 0.7152 + valor_b * 0.0722
            z = valor_r * 0.0193 + valor_g * 0.1192 + valor_b * 0.9505

            x /= 95.047
            y /= 100.000
            z /= 108.883

            fx = f(x)
            fy = f(y)
            fz = f(z)

            l = (116 * fy) - 16
            a = 500 * (fx - fy)
            b = 200 * (fy - fz)

            # Normalización para visualización
            canal_l[i, j] = [int(l * 255 / 100)] * 3
            canal_a[i, j] = [int(a + 128)] * 3
            canal_b[i, j] = [int(b + 128)] * 3

    # Mostrar los canales individuales
    cv2.imshow('Imagen original', imagen)
    cv2.imshow('Canal L', canal_l)
    cv2.imshow('Canal a', canal_a)
    cv2.imshow('Canal b', canal_b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

rgb_a_lab('skz.jpg')
