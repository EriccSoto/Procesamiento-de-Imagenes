import cv2
import numpy as np
# Cuanto más grande el kernel, más se encoge el objeto blanco.
def leer_imagen_binaria(ruta):
    # Lectura
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    _, binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    return binaria

def erosion_manual(imagen_binaria, tamaño_kernel=(3, 3)):
    # Tamaños
    alto = imagen_binaria.shape[0]
    ancho = imagen_binaria.shape[1]
    alto_kernel = tamaño_kernel[0]
    ancho_kernel = tamaño_kernel[1]
    padding_alto = alto_kernel // 2
    padding_ancho = ancho_kernel // 2

    # Relleno
    imagen_rellena = np.zeros((alto + 2 * padding_alto, ancho + 2 * padding_ancho), dtype=np.uint8)
    for y in range(alto):
        for x in range(ancho):
            imagen_rellena[y + padding_alto, x + padding_ancho] = imagen_binaria[y, x]

    # Resultado
    imagen_erosionada = np.zeros((alto, ancho), dtype=np.uint8)

    # Bucle
    for y in range(alto):
        for x in range(ancho):
            es_blanco = True
            for i in range(alto_kernel):
                for j in range(ancho_kernel):
                    pixel_vecino = imagen_rellena[y + i, x + j]
                    if pixel_vecino == 0:
                        es_blanco = False
            imagen_erosionada[y, x] = 255 if es_blanco else 0

    return imagen_erosionada

def erosion_con_opencv(imagen_binaria, tamaño_kernel=(5, 5)):
    # Librería
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tamaño_kernel)
    return cv2.erode(imagen_binaria, kernel, iterations=1)

if __name__ == "__main__":
    ruta_imagen = 'cuadro.png'
    imagen_binaria = leer_imagen_binaria(ruta_imagen)

    erosionada_manual = erosion_manual(imagen_binaria, (7, 7))
    erosionada_opencv = erosion_con_opencv(imagen_binaria, (2, 2))

    cv2.imshow('Original binaria', imagen_binaria)
    cv2.imshow('Erosion manual', erosionada_manual)
    cv2.imshow('Erosion con OpenCV', erosionada_opencv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
