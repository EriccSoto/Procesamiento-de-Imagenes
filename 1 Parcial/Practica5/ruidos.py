import cv2
import numpy as np
import random

# Función para añadir ruido de sal
def añadir_ruido_sal(imagen, porcentaje=0.02):
    filas, columnas = imagen.shape[:2]
    cantidad_sal = int(porcentaje * filas * columnas)

    for _ in range(cantidad_sal):
        i = random.randint(0, filas - 1)
        j = random.randint(0, columnas - 1)
        if len(imagen.shape) == 2:  # Escala de grises
            imagen[i, j] = 255
        else:
            imagen[i, j] = [255, 255, 255]
    return imagen

# Función para añadir ruido de pimienta
def añadir_ruido_pimienta(imagen, porcentaje=0.02):
    filas, columnas = imagen.shape[:2]
    cantidad_pimienta = int(porcentaje * filas * columnas)

    for _ in range(cantidad_pimienta):
        i = random.randint(0, filas - 1)
        j = random.randint(0, columnas - 1)
        if len(imagen.shape) == 2:
            imagen[i, j] = 0
        else:
            imagen[i, j] = [0, 0, 0]
    return imagen

# Función para añadir ruido de sal y pimienta
def añadir_ruido_sal_pimienta(imagen, porcentaje=0.02):
    filas, columnas = imagen.shape[:2]
    cantidad_ruido = int(porcentaje * filas * columnas)

    for _ in range(cantidad_ruido):
        i = random.randint(0, filas - 1)
        j = random.randint(0, columnas - 1)
        if random.random() < 0.5:
            if len(imagen.shape) == 2:
                imagen[i, j] = 255
            else:
                imagen[i, j] = [255, 255, 255]
        else:
            if len(imagen.shape) == 2:
                imagen[i, j] = 0
            else:
                imagen[i, j] = [0, 0, 0]
    return imagen

# Mostrar imágenes
def mostrar_imagen(nombre, imagen):
    cv2.imshow(nombre, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ruta de la imagen
ruta_imagen = "super.jpg"
imagen_original = cv2.imread(ruta_imagen)
mostrar_imagen("Imagen Original", imagen_original)

# Añadir ruido
imagen_sal = añadir_ruido_sal(imagen_original.copy(), 0.3)
imagen_pimienta = añadir_ruido_pimienta(imagen_original.copy(), 0.9)
imagen_sal_pimienta = añadir_ruido_sal_pimienta(imagen_original.copy(), 0.1)

# Mostrar imágenes con ruido
mostrar_imagen("Imagen con Ruido Sal", imagen_sal)
mostrar_imagen("Imagen con Ruido Pimienta", imagen_pimienta)
mostrar_imagen("Imagen con Ruido Sal y Pimienta", imagen_sal_pimienta)

# Guardar imágenes resultantes
cv2.imwrite("imagen_con_ruido_sal.jpg", imagen_sal)
cv2.imwrite("imagen_con_ruido_pimienta.jpg", imagen_pimienta)
cv2.imwrite("imagen_con_ruido_sal_pimienta.jpg", imagen_sal_pimienta)
