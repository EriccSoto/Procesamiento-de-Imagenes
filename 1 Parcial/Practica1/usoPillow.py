from PIL import Image
import os

#Abrir----------------------------------------------------------------------------

def abrir(ruta_imagen):
    
    if not os.path.exists(ruta_imagen):
        print(f"Error: El archivo no existe en {ruta_imagen}")
        return

    try:
        imagen = Image.open(ruta_imagen)
        print("Imagen Abierta")
        return imagen
    except Exception as e:
        print(f"Error al abrir la imagen: {e}")




#visualizar----------------------------------------------------------------------------
def visualizar_imagen_pillow(imagen):
    try:
        imagen.show()
    except Exception as e:
        print(f"Error al mostrar la imagen: {e}")

#Guardar----------------------------------------------------------------------------
def guardar_imagen_pillow(ruta_imagen, ruta_guardado):
 
    if not os.path.exists(ruta_imagen):
        print(f"Error: El archivo no existe en {ruta_imagen}")
        return

    try:
        imagen = Image.open(ruta_imagen)
        imagen.save(ruta_guardado)
        print(f"Imagen guardada en: {ruta_guardado}")
    except Exception as e:
        print(f"Error al guardar la imagen: {e}")


# Uso del código
ruta_original = "Practica1/Imagenes/tigre.jpg"
ruta_nueva = "Practica1/Imagenes/tigre3.jpg"
imagen = abrir(ruta_original)
visualizar_imagen_pillow(imagen)
guardar_imagen_pillow(ruta_original, ruta_nueva)
