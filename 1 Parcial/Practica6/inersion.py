import cv2
import numpy as np

# Función para modificar niveles de grises - Inversión
def inversion(imagen):
    L = 255
    resultado = np.zeros_like(imagen, dtype=np.uint8)
    filas, columnas, canales = imagen.shape
    
    for i in range(filas):
        for j in range(columnas):
            for k in range(canales):
                valor = L - int(imagen[i, j, k])
                resultado[i, j, k] = np.clip(valor, 0, 255)
    
    return resultado


def oscurecer(imagen):
    L = 255
    resultado = np.zeros_like(imagen, dtype=np.uint8)
    filas, columnas, canales = imagen.shape
    
    for i in range(filas):
        for j in range(columnas):
            for k in range(canales):
                valor = (int(imagen[i, j, k]) ** 2) // L
                resultado[i, j, k] = np.clip(valor, 0, 255)
    
    return resultado

# Función para aclarar la imagen
def aclarar(imagen):
    L = 255
    resultado = np.zeros_like(imagen, dtype=np.uint8)
    filas, columnas, canales = imagen.shape
    
    for i in range(filas):
        for j in range(columnas):
            for k in range(canales):
                valor = int(np.sqrt(L * int(imagen[i, j, k])))
                resultado[i, j, k] = np.clip(valor, 0, 255)
    
    return resultado

# Mostrar las imágenes
def mostrar_imagen(nombre, imagen):
    cv2.imshow(nombre, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ruta de la imagen
ruta_imagen = "super.jpg"

# Cargar imagen con cv2
imagen_original = cv2.imread(ruta_imagen)
if imagen_original is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta.")
else:
    mostrar_imagen("Imagen Original", imagen_original)

    # Aplicar transformaciones
    inversion_1 = inversion(imagen_original.copy())
    aclarar1 = aclarar(imagen_original.copy())
    oscurecer1 = oscurecer(imagen_original.copy())

    # Mostrar las imágenes con los cambios
    mostrar_imagen("Inversion", inversion_1)
    mostrar_imagen("Imagen Aclarada", aclarar1)
    mostrar_imagen("Imagen Oscurecida", oscurecer1)
