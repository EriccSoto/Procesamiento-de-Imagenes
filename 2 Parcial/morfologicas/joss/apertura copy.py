import cv2
import numpy as np

# Definir el elemento estructurante (kernel)
kernel = np.ones((5, 5), np.uint8)

# Obtener dimensiones del kernel
ker_h, ker_w = kernel.shape
pad_h, pad_w = ker_h // 2, ker_w // 2

# Función para la dilatación
def dilatacion(imagen1, kernel):
    img_h, img_w = imagen1.shape
    dilatada = np.zeros_like(imagen1)
    for i in range(pad_h, img_h - pad_h):
        for j in range(pad_w, img_w - pad_w):
            # Aplicar el kernel (región de interés)
            region = imagen1[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]
            # Asignar el valor máximo de la región al píxel resultante
            dilatada[i, j] = np.max(region * kernel)
    return dilatada

# Función para la erosión
def erosion(imagen1, kernel):
    img_h, img_w = imagen1.shape
    erosionada = np.zeros_like(imagen1)
    for i in range(pad_h, img_h - pad_h):
        for j in range(pad_w, img_w - pad_w):
            # Aplicar el kernel (región de interés)
            region = imagen1[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]
            # Asignar el valor mínimo de la región al píxel resultante
            erosionada[i, j] = np.min(region * kernel)
    return erosionada


if __name__ == '__main__':
    imagen1= cv2.imread('imagen1.png', cv2.IMREAD_GRAYSCALE)
    if imagen1 is None:
        print("Error: No se pudo cargar la imagen1.")
    else:
        # Aplicar erosión seguida de dilatación (apertura)
        erosionada = erosion(imagen1, kernel)
        apertura = dilatacion(erosionada, kernel)

        # Mostrar las imágenes
        cv2.imshow('imagen1 Original', imagen1)
        cv2.imshow('imagen1 Erosionada', erosionada)
        cv2.imshow('imagen1 con Apertura', apertura)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

  