import cv2
import os

#Abrir----------------------------------------------------------------------------
def abrir(ruta_imagen):
    if not os.path.exists(ruta_imagen):
        print(f"Error: El archivo no existe en {ruta_imagen}")
        return None

    try:
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print("Error: No se pudo leer la imagen.")
            return None
        print("Imagen Abierta")
        return imagen
    except Exception as e:
        print(f"Error al abrir la imagen: {e}")
        return None


#visualizar----------------------------------------------------------------------------
def visualizar_imagen_cv2(imagen):
    if imagen is not None:
        cv2.imshow("Imagen", imagen)
        cv2.waitKey(0)  # Espera a que se presione una tecla
        cv2.destroyAllWindows()
    else:
        print("No se pudo visualizar la imagen.")

#Guardar----------------------------------------------------------------------------
def guardar_imagen_cv2(ruta_imagen, ruta_guardado):
    if not os.path.exists(ruta_imagen):
        print(f"Error: El archivo no existe en {ruta_imagen}")
        return

    try:
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print("Error: No se pudo leer la imagen.")
            return
        cv2.imwrite(ruta_guardado, imagen)
        print(f"Imagen guardada en: {ruta_guardado}")
    except Exception as e:
        print(f"Error al guardar la imagen: {e}")

# Uso del código
ruta_original = "Practica1/Imagenes/tigre.jpg"
ruta_nueva = "Practica1/Imagenes/tigre2.jpg"
imagen = abrir(ruta_original)
visualizar_imagen_cv2(imagen)
guardar_imagen_cv2(ruta_original, ruta_nueva)
