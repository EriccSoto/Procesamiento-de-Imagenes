import cv2
import numpy as np

imagen = cv2.imread('ooo.png', cv2.IMREAD_GRAYSCALE)
if imagen is None:
    print("Error al cargar la imagen.")
    exit()
#-----------------------------------------------------------------------------
kernel = np.ones((5, 5), np.uint8)
img_h, img_w = imagen.shape
ker_h, ker_w = kernel.shape
pad_h, pad_w = ker_h // 2, ker_w // 2

# Función para la dilatación
def dilatacion(imagen, kernel):
    dilatada = np.zeros_like(imagen)
    for i in range(pad_h, img_h - pad_h):
        for j in range(pad_w, img_w - pad_w):
            # Aplicar el kernel (región de interés)
            region = imagen[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]
            # Asignar el valor máximo de la región al píxel resultante
            dilatada[i, j] = np.max(region * kernel)
    return dilatada

# Función para la erosión
def erosion(imagen, kernel):
    erosionada = np.zeros_like(imagen)
    for i in range(pad_h, img_h - pad_h):
        for j in range(pad_w, img_w - pad_w):
            # Aplicar el kernel (región de interés)
            region = imagen[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]
            # Asignar el valor mínimo de la región al píxel resultante
            erosionada[i, j] = np.min(region * kernel)
    return erosionada

# Aplicar dilatación seguida de erosión (cierre)
dilatada = dilatacion(imagen, kernel)
cierre_manual = erosion(dilatada, kernel)

cv2.imshow('Imagen Original', imagen)
cv2.imshow('Imagen dilatada', dilatada)
cv2.imshow('Imagen con Cierre', cierre_manual)
cv2.waitKey(0)
cv2.destroyAllWindows()
