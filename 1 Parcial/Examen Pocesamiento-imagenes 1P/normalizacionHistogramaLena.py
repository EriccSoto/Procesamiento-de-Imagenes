import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


#--------- Abrir Imagen ------------------------------
def abrir(ruta_imagen):
    if not os.path.exists(ruta_imagen):
        print(f"Error: El archivo no existe en {ruta_imagen}")
        return None

    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_UNCHANGED) 
    if imagen is None:
        print("Error: No se pudo leer la imagen.")
        return None
    print("Imagen Abierta")
    return imagen
#------------------------------------------------------

#--------- Características de la Imagen ------------------------------
def caracteristicas(imagen):
    print("\nTipo de imagen:", type(imagen))
    print("Dimensiones (alto, ancho, canales):", imagen.shape)
#------------------------------------------------------

#--------- Mostrar Imágenes ------------------------------
def mostrar(imagen_original, imagen_modificada, leyenda):
    print("Características de la imagen original:")
    caracteristicas(imagen_original)
    print("Características de la imagen modificada:")
    caracteristicas(imagen_modificada)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    if (len(imagen_original.shape) == 2):
        axes[0].imshow(imagen_original, cmap="gray")
    else:
        axes[0].imshow(cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB))

    if (len(imagen_modificada.shape) == 2):
        axes[1].imshow(imagen_modificada, cmap="gray")
    else:
        axes[1].imshow(cv2.cvtColor(imagen_modificada, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")
    axes[1].set_title(leyenda)
    axes[1].axis("off")

    plt.show()
#------------------------------------------------------

#--------- Calcular Histograma ------------------------------
def calcular_histograma(imagen, canal=None):
    if len(imagen.shape) == 2:
        histograma = np.zeros(256)
        for valor in imagen.ravel():
            valor = int(valor)  # Convertir el valor a entero
            histograma[valor] += 1
    else:
        if canal is not None:
            histograma = np.zeros(256)
            for valor in imagen[:, :, canal].ravel():
                valor = int(valor)  # Convertir el valor a entero
                histograma[valor] += 1
        else:
            histograma = np.zeros((256, 3))
            for i in range(3):
                for valor in imagen[:, :, i].ravel():
                    valor = int(valor)  # Convertir el valor a entero
                    histograma[valor, i] += 1
                    
    return histograma
#------------------------------------------------------
#--------- Mostrar Histograma de Imagen Original ------------------------------
def mostrarHistogramaOriginal(imagen):
    plt.figure(figsize=(10, 6))

    if len(imagen.shape) == 2:
        histograma = calcular_histograma(imagen)
        plt.plot(histograma, color="black", label="Escala de Grises")
    else:
        colores = ('b', 'g', 'r')
        for i, c in enumerate(colores):
            histograma = calcular_histograma(imagen, canal=i)
            plt.plot(histograma, color=c, label=f'Canal {c.upper()}')

    plt.title("Histograma de la Imagen Original")
    plt.xlabel("Intensidad de píxel")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.show()
#------------------------------------------------------

#--------- Mostrar Histograma de Imagen Umbralizada ------------------------------
def mostrarHistogramaUmbralizada(imagen, leyenda):
    if imagen is None:
        print("Error: La imagen no se ha cargado correctamente.")
        return

    if len(imagen.shape) == 2:
        plt.figure(figsize=(10, 6))
        histograma = calcular_histograma(imagen)
        plt.plot(histograma, color="black", label="Escala de Grises")
        plt.title(leyenda)
        plt.xlabel("Intensidad de píxel")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.show()
    else:
        colores = ('b', 'g', 'r')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, c in enumerate(colores):
            histograma = calcular_histograma(imagen, canal=i)
            axes[i].plot(histograma, color=c)
            axes[i].set_title(f"Histograma Canal {c.upper()}")
            axes[i].set_xlabel("Intensidad de píxel")
            axes[i].set_ylabel("Frecuencia")

        plt.tight_layout()
        plt.show()
#------------------------------------------------------

#--------- Modificar Histograma de Imagen ------------------------------
def modificar_histograma(imagen, n, N):
    if len(imagen.shape) == 2:
        imagen = imagen.astype(np.float32)
        m = np.min(imagen)
        M = np.max(imagen)
        
        imagen_modificada = n + ((imagen - m) * (N - n)) / (M - m)
        
    else:
        imagen_modificada = np.zeros_like(imagen, dtype=np.float32)
        for i in range(3):
            canal = imagen[:, :, i].astype(np.float32)
            m = np.min(canal)
            M = np.max(canal)
            
            imagen_modificada[:, :, i] = n + ((canal - m) * (N - n)) / (M - m)
    
    imagen_modificada = imagen_modificada.astype(np.uint8)
    
    mostrar(imagen, imagen_modificada, "Imagen con Histograma Normalizado")
    mostrarHistogramaOriginal(imagen)
    mostrarHistogramaUmbralizada(imagen_modificada, "Histograma Normalizado")

    return imagen_modificada
#------------------------------------------------------


#--------- EJECUCIÓN ------------------------------

ruta_original = "Baboon.png" 
imglena = abrir(ruta_original) #Imagen a Color
imglenaBN = cv2.cvtColor(imglena, cv2.COLOR_BGR2GRAY)#imagen a BN

plt.imshow(cv2.cvtColor(imglena, cv2.COLOR_BGR2RGB))
plt.title("Imagen Original")
plt.axis('off') 
plt.show()
mostrarHistogramaOriginal(imglena)



# Modificación de histograma ---------------------------------------------

n = 50  # Valor mínimo deseado
N = 160 # Valor máximo deseado

imgLena_modificada = modificar_histograma(imglena, n, N)
n = 0  # Valor mínimo deseado
N = 6 # Valor máximo deseado

imgLena_modificada = modificar_histograma(imglenaBN, n, N)

