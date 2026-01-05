import cv2
import os
import numpy as np

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


#caracteristicas----------------------------------------------------------------------------
def caracteristicas(imagen):
    print("\nTipo de imagen", type(imagen))
    print("\nImagen (alto, ancho, canales)", (imagen.shape))
        
  


def rotacion(imagen, angulo):
    # Convertir el ángulo de grados a radianes
    theta = np.radians(angulo)
    
    # Obtener el tamaño original de la imagen
    alto, ancho = imagen.shape[:2]
    
    # Calcular el nuevo tamaño maximo que puede ocupar la imagen rotada
    nuevo_ancho = int(ancho * abs(np.cos(theta)) + alto * abs(np.sin(theta)))
    nuevo_alto = int(alto * abs(np.cos(theta)) + ancho * abs(np.sin(theta)))
    
    # Crear una imagen con el nuevo tamaño con fondo negro
    imagen_modificada = np.zeros((nuevo_alto, nuevo_ancho, 3), dtype=np.uint8)
    
    # Calcular el nuevo centro de la imagen
    nuevo_cx, nuevo_cy = nuevo_ancho // 2, nuevo_alto // 2
    cx, cy = ancho // 2, alto // 2
    
    # Aplicar la transformacion de rotacion a cada pxel
    for i in range(alto):
        for j in range(ancho):
            # Trasladar al origen
            x = j - cx
            y = i - cy
            
            # Aplicar las ecuaciones de rotacin
            x_rot = int(x * np.cos(theta) - y * np.sin(theta) + nuevo_cx)
            y_rot = int(y * np.cos(theta) + x * np.sin(theta) + nuevo_cy)
            
            if 0 <= x_rot < nuevo_ancho and 0 <= y_rot < nuevo_alto:
                imagen_modificada[y_rot, x_rot] = imagen[i, j]
    
    # Mostrar caracteristicas de las imagenes
    caracteristicas(imagen)
    caracteristicas(imagen_modificada)
    
    # Mostrar la imagen rotada
    cv2.imshow("Imagen Rotada", imagen_modificada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



ruta_original = "Baboon.png"
mono = abrir(ruta_original)
rotacion(mono, 120)