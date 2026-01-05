import cv2
import numpy as np

def leer_imagen_binaria(ruta):
    # Lectura
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    _, binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    return binaria

def cerradura_manual(imagen_binaria, tamaño_kernel=(3, 3)):
    # Dilatación manual
    dilatada = dilatacion_manual(imagen_binaria, tamaño_kernel)
    cv2.imshow('Original binaria', imagen_binaria)
    cv2.imshow('paso 1 dilatada', dilatada)
    
    # Erosión manual sobre la imagen dilatada
    erosionada = erosion_manual(dilatada, tamaño_kernel)
    cv2.imshow('paso 2 erosionada', erosionada)
    return erosionada

def cerradura_con_opencv(imagen_binaria, tamaño_kernel=(5, 5)):
    # Librería
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tamaño_kernel)
    return cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel)

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

if __name__ == "__main__":
    ruta_imagen = 'cuadro.png'
    imagen_binaria = leer_imagen_binaria(ruta_imagen)

    cerradura_manual = cerradura_manual(imagen_binaria, (7, 7))
    cerradura_opencv = cerradura_con_opencv(imagen_binaria, (2, 2))

    cv2.imshow('Cerradura manual', cerradura_manual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Cerradura con OpenCV', cerradura_opencv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



