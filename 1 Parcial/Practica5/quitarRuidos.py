import cv2
import numpy as np

# Función para obtener el promedio de vecinos
def obtener_promedio_vecinos(imagen, i, j):
    vecinos = []
    filas, columnas = imagen.shape[:2]

    for x in range(-1, 2):
        for y in range(-1, 2):
            nx, ny = i + x, j + y
            # Verificar que el vecino está dentro de los límites
            if 0 <= nx < filas and 0 <= ny < columnas and (nx != i or ny != j):
                vecinos.append(imagen[nx, ny])

    if len(imagen.shape) == 2:
        return int(np.mean(vecinos))
    else:
        return [int(np.mean([vecino[c] for vecino in vecinos])) for c in range(3)]


# Función para eliminar ruido de sal
def eliminar_ruido_sal(imagen):
    filas, columnas = imagen.shape[:2]
    imagen_filtrada = imagen.copy()

    for i in range(1, filas - 1):
        for j in range(1, columnas - 1):
            if len(imagen.shape) == 2:  # Escala de grises
                if imagen[i, j] == 255:  # Detección de sal
                    imagen_filtrada[i, j] = obtener_promedio_vecinos(imagen, i, j)
            else:  # Imagen a color
                if (imagen[i, j] == [255, 255, 255]).all():
                    imagen_filtrada[i, j] = obtener_promedio_vecinos(imagen, i, j)

    return imagen_filtrada


# Función para eliminar ruido de pimienta
def eliminar_ruido_pimienta(imagen):
    filas, columnas = imagen.shape[:2]
    imagen_filtrada = imagen.copy()

    for i in range(1, filas - 1):
        for j in range(1, columnas - 1):
            if len(imagen.shape) == 2:  # Escala de grises
                if imagen[i, j] == 0:  # Detección de pimienta
                    imagen_filtrada[i, j] = obtener_promedio_vecinos(imagen, i, j)
            else:  # Imagen a color
                if (imagen[i, j] == [0, 0, 0]).all():
                    imagen_filtrada[i, j] = obtener_promedio_vecinos(imagen, i, j)

    return imagen_filtrada


# Función para eliminar ruido de sal y pimienta
def eliminar_ruido_sal_pimienta(imagen):
    filas, columnas = imagen.shape[:2]
    imagen_filtrada = imagen.copy()

    for i in range(1, filas - 1):
        for j in range(1, columnas - 1):
            if len(imagen.shape) == 2:  # Escala de grises
                if imagen[i, j] == 255 or imagen[i, j] == 0:
                    imagen_filtrada[i, j] = obtener_promedio_vecinos(imagen, i, j)
            else:  # Imagen a color
                if (imagen[i, j] == [255, 255, 255]).all() or (imagen[i, j] == [0, 0, 0]).all():
                    imagen_filtrada[i, j] = obtener_promedio_vecinos(imagen, i, j)

    return imagen_filtrada


# Ruta de la imagen
ruta_imagen = "super.jpg"
vibora1 = "vibora.jpg"
# Cargar imagen con cv2
imagen_original = cv2.imread(ruta_imagen)
vibora = cv2.imread(vibora1)

# Eliminar diferentes tipos de ruido
imagen_sal_filtrada = eliminar_ruido_sal(imagen_original.copy())
imagen_pimienta_filtrada = eliminar_ruido_pimienta(imagen_original.copy())
imagen_sal_pimienta_filtrada = eliminar_ruido_sal_pimienta(vibora)

# Mostrar las imágenes filtradas
def mostrar_imagen(nombre, imagen):
    cv2.imshow(nombre, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Mostrar las imágenes sin ruido
mostrar_imagen("Imagen Filtrada Sal", imagen_sal_filtrada)
mostrar_imagen("Imagen Filtrada Pimienta", imagen_pimienta_filtrada)
mostrar_imagen("Imagen Filtrada Sal y Pimienta", imagen_sal_pimienta_filtrada)
        