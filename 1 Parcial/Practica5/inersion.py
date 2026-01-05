import cv2
import numpy as np

# Función para modificar niveles de grises - Inversión
def inversion(imagen):
    imagen = imagen.astype(np.int32)
    L = 255
    # Aplicar fórmula p = L - m / p=255-30=225
    imagen = L - imagen
    imagen = np.clip(imagen, 0, 255).astype(np.uint8)
    return imagen


def oscurecer(imagen):
    imagen = imagen.astype(np.int32)
    L = 255
    # Aplicar fórmula p = (m^2) / L ejemplo p=30^2/255=3 oscurece
    imagen = (imagen ** 2) // L
    imagen = np.clip(imagen, 0, 255).astype(np.uint8)
    return imagen


# Función para aclarar la imagen
def aclarar(imagen):
    imagen = imagen.astype(np.int32)
    L = 255
    # Aplicar fórmula p = √(L * m)   ejemplo  P=√(255*30)=87 aclara
    imagen = np.sqrt(L * imagen)
    imagen = np.clip(imagen, 0, 255).astype(np.uint8)
    return imagen


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
