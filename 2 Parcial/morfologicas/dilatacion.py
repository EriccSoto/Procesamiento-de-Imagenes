import cv2
import numpy as np
#un kernel grande, la dilatación se vuelve más agresiva
def leer_imagen_binaria(ruta):
    # Lectura
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    _, binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    return binaria

def dilatacion_manual(imagen_binaria, tamaño_kernel=(3, 3)):
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
    imagen_dilatada = np.zeros((alto, ancho), dtype=np.uint8)

    # Bucle
    for y in range(alto):
        for x in range(ancho):
            es_negro = True
            for i in range(alto_kernel):
                for j in range(ancho_kernel):
                    pixel_vecino = imagen_rellena[y + i, x + j]
                    if pixel_vecino == 255:
                        es_negro = False
            imagen_dilatada[y, x] = 0 if es_negro else 255

    return imagen_dilatada

def dilatacion_con_opencv(imagen_binaria, tamaño_kernel=(5, 5)):
    # Librería
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tamaño_kernel)
    return cv2.dilate(imagen_binaria, kernel, iterations=1)

if __name__ == "__main__":
    ruta_imagen = 'cuadro2.png'
    imagen_binaria = leer_imagen_binaria(ruta_imagen)

    dilatada_manual = dilatacion_manual(imagen_binaria, (7, 7))
    dilatada_opencv = dilatacion_con_opencv(imagen_binaria, (2, 2))

    cv2.imshow('Original binaria', imagen_binaria)
    cv2.imshow('Dilatacion manual', dilatada_manual)
    cv2.imshow('Dilatacion con OpenCV', dilatada_opencv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
