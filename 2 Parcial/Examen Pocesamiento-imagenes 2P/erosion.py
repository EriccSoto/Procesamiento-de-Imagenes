import cv2
import numpy as np

def leer_imagen_binaria(ruta):
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    _, binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    return binaria

def erosion_manual(imagen_binaria, tamaño_kernel=(3, 3)):
    alto, ancho = imagen_binaria.shape
    k_alto, k_ancho = tamaño_kernel
    p_alto, p_ancho = k_alto // 2, k_ancho // 2

    # Relleno con ceros (negro)
    imagen_rellena = np.zeros((alto + 2 * p_alto, ancho + 2 * p_ancho), dtype=np.uint8)
    imagen_rellena[p_alto:alto + p_alto, p_ancho:ancho + p_ancho] = imagen_binaria

    imagen_erosionada = np.zeros((alto, ancho), dtype=np.uint8)

    for y in range(alto):
        for x in range(ancho):
            # Si el píxel está en los bordes, lo dejamos en blanco directamente
            if y < p_alto or y >= alto - p_alto or x < p_ancho or x >= ancho - p_ancho:
                imagen_erosionada[y, x] = 255
                continue

            # Revisamos el kernel
            es_blanco = True
            for i in range(k_alto):
                for j in range(k_ancho):
                    if imagen_rellena[y + i, x + j] == 0:
                        es_blanco = False
                        break
                if not es_blanco:
                    break

            imagen_erosionada[y, x] = 255 if es_blanco else 0

    return imagen_erosionada

def erosion_con_opencv(imagen_binaria, tamaño_kernel=(2, 2)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tamaño_kernel)
    return cv2.erode(imagen_binaria, kernel, iterations=1)

if __name__ == "__main__":
    ruta_imagen = 'Avion.PNG'
    imagen_binaria = leer_imagen_binaria(ruta_imagen)

    erosionada_manual = erosion_manual(imagen_binaria, (7, 7))
    erosionada_opencv = erosion_con_opencv(imagen_binaria, (5, 5))

    cv2.imshow('Original binaria', imagen_binaria)
    cv2.imshow('Erosion manual (bordes blancos)', erosionada_manual)
    cv2.imshow('Erosion con OpenCV', erosionada_opencv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
