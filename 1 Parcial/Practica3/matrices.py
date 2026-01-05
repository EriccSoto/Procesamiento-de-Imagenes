import cv2
import os
import numpy as np
import random

#Crear----------------------------------------------------------------------------+
def crearImg():
    # Crear imagen negra de 200x200 con 3 canales (RGB)
    N, M, C = 200, 200, 3
    imn_n = np.zeros((N, M, C), dtype=np.uint8)
    return imn_n

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
    #Alto: Número de filas (píxeles en la altura de la imagen).
    #Ancho: Número de columnas (píxeles en la anchura de la imagen).
    #Canales: Número de canales de color (generalmente 3 para RGB o 4 para RGBA).
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
    


def mostrarV2(imagen_original1,imagen_original2, imagen_modificada, leyenda):
    
    if imagen_original1 is None or imagen_original2 is None or imagen_modificada is None:
        print("Error: Una de las imágenes no se ha cargado correctamente.")
        return

    # Mostrar las características de ambas imágenes
    print("Características de la imagen original 1:")
    caracteristicas(imagen_original1)
    print("Características de la imagen original 2:")
    caracteristicas(imagen_original2)
    print("Características de la imagen modificada:")
    caracteristicas(imagen_modificada)

    # Mostrar ambas imágenes en ventanas separadas
    cv2.imshow('Imagen Original 1', imagen_original1)
    cv2.imshow('Imagen Original 2', imagen_original2)
    cv2.imshow(leyenda, imagen_modificada)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if imagen_original1 is None or imagen_original2 is None or imagen_modificada is None:
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


#Brillo--------------------------------------------------------------------------------    


def brillo(imagen, valor_brillo):
    print(imagen.shape)
    alto, ancho, ncolores = imagen.shape
    print(f"Ancho: {ancho}, Alto: {alto}, Número de colores: {ncolores}")
    
    imagenBrillo = np.zeros((int(alto), int(ancho), ncolores), dtype='uint8')
    
    # Iterar sobre cada píxel y modificar su brillo
    for y in range(alto):
        for x in range(ancho):
            for c in range(ncolores):  
                nuevo_valor = int(imagen[y, x, c]) + valor_brillo 
                if nuevo_valor > 255:
                    imagenBrillo[y, x, c] = 255
                elif nuevo_valor < 0:
                    imagenBrillo[y, x, c] = 0
                else:
                    imagenBrillo[y, x, c] = nuevo_valor
    mostrar(imagen, imagenBrillo, "Ajuste de brillo")


#suma--------------------------------------------------------------------------------------


def redimensionar(imagen, alto_destino, ancho_destino):
    alto, ancho, ncolores = imagen.shape
    imagen_redimensionada = np.zeros((alto_destino, ancho_destino, ncolores), dtype='uint8')
    
    factor_y = alto / alto_destino
    factor_x = ancho / ancho_destino
    
    for y in range(alto_destino):
        for x in range(ancho_destino):
            y_original = int(y * factor_y)
            x_original = int(x * factor_x)
            
            imagen_redimensionada[y, x, :] = imagen[y_original, x_original, :]
    
    return imagen_redimensionada


#suma--------------------------------------------------------------------------------------


def suma(imagen1, imagen2):
    alto1, ancho1, _ = imagen1.shape
    alto2, ancho2, _ = imagen2.shape
    
    # Redimensiona la imagen
    if alto1 > alto2 or ancho1 > ancho2:
        imagen1 = redimensionar(imagen1, alto2, ancho2)
    elif alto2 > alto1 or ancho2 > ancho1:
        imagen2 = redimensionar(imagen2, alto1, ancho1)
    
    alto_comun, ancho_comun, _ = imagen1.shape
    
    imagen_resultado = np.zeros((alto_comun, ancho_comun, 3), dtype='uint8')
    
    for y in range(alto_comun):
        for x in range(ancho_comun):
            for c in range(3):
                valor = imagen1[y, x, c] + imagen2[y, x, c]
                
                if valor > 255:
                    imagen_resultado[y, x, c] = 255
                elif valor < 0:
                    imagen_resultado[y, x, c] = 0
                else:
                    imagen_resultado[y, x, c] = valor
    
    mostrarV2(imagen1, imagen2, imagen_resultado, "Suma")


#resta--------------------------------------------------------------------------------------

def resta(imagen1, imagen2):
    alto1, ancho1, _ = imagen1.shape
    alto2, ancho2, _ = imagen2.shape
    
    # Redimensiona la imagen
    if alto1 > alto2 or ancho1 > ancho2:
        imagen1 = redimensionar(imagen1, alto2, ancho2)
    elif alto2 > alto1 or ancho2 > ancho1:
        imagen2 = redimensionar(imagen2, alto1, ancho1)
    
    alto_comun, ancho_comun, _ = imagen1.shape
    
    imagen_resultado = np.zeros((alto_comun, ancho_comun, 3), dtype='uint8')
    
    for y in range(alto_comun):
        for x in range(ancho_comun):
            for c in range(3):
                valor = imagen1[y, x, c] - imagen2[y, x, c]
                
                if valor > 255:
                    imagen_resultado[y, x, c] = 255
                elif valor < 0:
                    imagen_resultado[y, x, c] = 0
                else:
                    imagen_resultado[y, x, c] = valor
    
    mostrarV2(imagen1, imagen2, imagen_resultado, "Resta")


#multiplicacion--------------------------------------------------------------------------------------

def multiplicacion(imagen1, imagen2):
    alto1, ancho1, _ = imagen1.shape
    alto2, ancho2, _ = imagen2.shape
    
    # Redimensiona la imagen
    if alto1 > alto2 or ancho1 > ancho2:
        imagen1 = redimensionar(imagen1, alto2, ancho2)
    elif alto2 > alto1 or ancho2 > ancho1:
        imagen2 = redimensionar(imagen2, alto1, ancho1)
    
    alto_comun, ancho_comun, _ = imagen1.shape
    
    imagen_resultado = np.zeros((alto_comun, ancho_comun, 3), dtype='uint8')
    
    for y in range(alto_comun):
        for x in range(ancho_comun):
            for c in range(3):
                valor =(int( imagen1[y, x, c]) * int(imagen2[y, x, c]))
                #print("antes")
                #print(f"{( imagen1[y, x, c])} x {(imagen2[y, x, c])} = {valor}")
                if valor > 255:
                    imagen_resultado[y, x, c] = 255
                elif valor < 0:
                    imagen_resultado[y, x, c] = 0
                else:
                    imagen_resultado[y, x, c] = valor
                #print("despues",imagen_resultado[y, x, c])
    mostrarV2(imagen1, imagen2, imagen_resultado, "Multiplicacion")


#division--------------------------------------------------------------------------------------

def division(imagen1, imagen2):
    alto1, ancho1, _ = imagen1.shape
    alto2, ancho2, _ = imagen2.shape
    
    # Redimensiona la imagen
    if alto1 > alto2 or ancho1 > ancho2:
        imagen1 = redimensionar(imagen1, alto2, ancho2)
    elif alto2 > alto1 or ancho2 > ancho1:
        imagen2 = redimensionar(imagen2, alto1, ancho1)
    
    alto_comun, ancho_comun, _ = imagen1.shape
    
    imagen_resultado = np.zeros((alto_comun, ancho_comun, 3), dtype='uint8')
    
    for y in range(alto_comun):
        for x in range(ancho_comun):
            for c in range(3):
                valor =(( imagen1[y, x, c]) / (imagen2[y, x, c]))
                
                if valor > 255:
                    imagen_resultado[y, x, c] = 255
                elif valor < 0:
                    imagen_resultado[y, x, c] = 0
                else:
                    imagen_resultado[y, x, c] = valor
    
    mostrarV2(imagen1, imagen2, imagen_resultado, "Division")

#and--------------------------------------------------------------------------------------

def andd(imagen1, imagen2):
    alto1, ancho1, _ = imagen1.shape
    alto2, ancho2, _ = imagen2.shape
    
    # Redimensiona la imagen
    if alto1 > alto2 or ancho1 > ancho2:
        imagen1 = redimensionar(imagen1, alto2, ancho2)
    elif alto2 > alto1 or ancho2 > ancho1:
        imagen2 = redimensionar(imagen2, alto1, ancho1)
    
    alto_comun, ancho_comun, _ = imagen1.shape
    
    imagen_resultado = np.zeros((alto_comun, ancho_comun, 3), dtype='uint8')
    
    for y in range(alto_comun):
        for x in range(ancho_comun):
            for c in range(3):
                imagen_resultado[y, x, c]  =(( imagen1[y, x, c]) & (imagen2[y, x, c]))
                
    mostrarV2(imagen1, imagen2, imagen_resultado, "And")

#or--------------------------------------------------------------------------------------

def orr(imagen1, imagen2):
    alto1, ancho1, _ = imagen1.shape
    alto2, ancho2, _ = imagen2.shape
    
    # Redimensiona la imagen
    if alto1 > alto2 or ancho1 > ancho2:
        imagen1 = redimensionar(imagen1, alto2, ancho2)
    elif alto2 > alto1 or ancho2 > ancho1:
        imagen2 = redimensionar(imagen2, alto1, ancho1)
    
    alto_comun, ancho_comun, _ = imagen1.shape
    
    imagen_resultado = np.zeros((alto_comun, ancho_comun, 3), dtype='uint8')
    
    for y in range(alto_comun):
        for x in range(ancho_comun):
            for c in range(3):
                imagen_resultado[y, x, c] =(( imagen1[y, x, c]) | (imagen2[y, x, c]))
    
    mostrarV2(imagen1, imagen2, imagen_resultado, "Or")

#not--------------------------------------------------------------------------------    

def nott(imagen):
    print(imagen.shape)
    alto, ancho, ncolores = imagen.shape
    print(f"Ancho: {ancho}, Alto: {alto}, Número de colores: {ncolores}")
    
    imagenNot = np.zeros((int(alto), int(ancho), ncolores), dtype='uint8')
    
    for y in range(alto):
        for x in range(ancho):
            for c in range(ncolores):
                imagenNot[y, x, c]= ~imagen[y, x, c]
    
    # Función para mostrar las imágenes (suponiendo que existe)
    mostrar(imagen, imagenNot, "Not")

    
ruta_original = "Imagenes/skz.jpg"
img_skz = abrir(ruta_original)
ruta_original = "Imagenes/blink.jpg"
imgBlink = abrir(ruta_original)
#copia(img_skz)
#inversion(img_skz)
#brillo(img_skz,70)
#suma(imgBlink,img_skz)
#resta(imgBlink,img_skz)
#multiplicacion(imgBlink,img_skz)
division(imgBlink,img_skz)
#andd(imgBlink,img_skz)
#orr(imgBlink,img_skz)
#nott(imgBlink)











