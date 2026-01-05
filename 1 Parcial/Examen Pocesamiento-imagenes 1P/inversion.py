import cv2
import os
import numpy as np
import random

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


#mostrar----------------------------------------------------------------------------

def mostrar(imagen_original, imagen_modificada, leyenda):
    
    if imagen_original is None or imagen_modificada is None:
        print("Error: Una de las imágenes no se ha cargado correctamente.")
        return
    # Mostrar las características de ambas imágenes
    print("Características de la imagen original:")
    caracteristicas(imagen_original)
    print("Características de la imagen modificada:")
    caracteristicas(imagen_modificada)

    # Mostrar ambas imágenes en ventanas separadas
    cv2.imshow('Imagen Original', imagen_original)
    cv2.imshow(leyenda, imagen_modificada)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if imagen_original is None or imagen_modificada is None:
        print("Error: Una de las imágenes no se ha cargado correctamente.")
        return
    

#copia--------------------------------------------------------------------------------    
   
def copia(imagen):
    print(imagen.shape)
    # Obtener dimensiones
    alto, ancho, ncolores = imagen.shape
    print(f"Ancho: {ancho}, Alto: {alto}, Número de colores: {ncolores}")

    # Crear copia de la imagen
    imagenCopia = np.zeros((alto, ancho, ncolores), dtype='uint8')
    
    for y in range(alto):
        for x in range(ancho):
            imagenCopia[y, x, :] = imagen[y, x, :]

    mostrar(imagen, imagenCopia, "Copia")

#inversion--------------------------------------------------------------------------------    
   
def inversion(imagen):
    print(imagen.shape)
    # Obtener dimensiones
    alto, ancho, ncolores = imagen.shape
    print(f"Ancho: {ancho}, Alto: {alto}, Número de colores: {ncolores}")
    # Genera copia de mismas dimensiones
    imgInversa = np.zeros((int(alto), int(ancho), ncolores),dtype='uint8')
    L = [255, 255, 255]#matriz L para que se resten los colores y se inviertan
    for y in range(alto):
        for x in range(ancho):
            imgInversa[y, x, :] = L - imagen[y, x, :]

    mostrar(imagen, imgInversa, "Inversion de Color")


    
ruta_original = "Baboon.png" 
img_mono = abrir(ruta_original)
inversion(img_mono)

