#--------- Importar Librerías ------------------------------
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


#--------- Funciones ------------------------------

#--------- Crear Imagen ------------------------------
def crearImg():
    return np.zeros((200, 200, 3), dtype=np.uint8)
#------------------------------------------------------

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

#--------- Binarización de Imagen ------------------------------
def Binarizado(imagen, umbral):
    if len(imagen.shape) == 2:
        imagen_binarizada = np.where(imagen >= umbral, 255, 0).astype(np.uint8)
    else:
        imagen_binarizada = np.zeros_like(imagen)
        imagen_binarizada[:, :, :] = np.where(imagen[:, :, :] >= umbral, 255, 0)

    mostrar(imagen, imagen_binarizada, "Imagen Binarizada")
    mostrarHistogramaOriginal(imagen)
    mostrarHistogramaUmbralizada(imagen_binarizada, "Imagen Binarizada")

    return imagen_binarizada
#------------------------------------------------------

#--------- Umbralización Nivelada ------------------------------
def umbralizacion_nivelada(imagen, niveles, umbrales, valores):
    if len(imagen.shape) == 2:
        imagen_umbr = np.zeros_like(imagen)
        for x in range(imagen.shape[0]):
            for y in range(imagen.shape[1]):
                valor = imagen[x, y]
                for i in range(niveles - 1):
                    if umbrales[i] < valor <= umbrales[i + 1]:
                        imagen_umbr[x, y] = valores[i + 1]
                        break
                    elif valor <= umbrales[0]:
                        imagen_umbr[x, y] = valores[0]
                        break
                    elif valor > umbrales[-1]:
                        imagen_umbr[x, y] = valores[-1]
                        break
    else:
        imagen_umbr = np.zeros_like(imagen)
        for i in range(3):
            for x in range(imagen.shape[0]):
                for y in range(imagen.shape[1]):
                    valor = imagen[x, y, i]
                    for j in range(niveles - 1):
                        if umbrales[j] < valor <= umbrales[j + 1]:
                            imagen_umbr[x, y, i] = valores[j + 1]
                            break
                        elif valor <= umbrales[0]:
                            imagen_umbr[x, y, i] = valores[0]
                            break
                        elif valor > umbrales[-1]:
                            imagen_umbr[x, y, i] = valores[-1]
                            break
    
    mostrar(imagen, imagen_umbr, "Imagen con N umbrales")
    mostrarHistogramaOriginal(imagen)
    mostrarHistogramaUmbralizada(imagen_umbr, "Hisstograma con N umbrales")

    return imagen_umbr
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
    
    mostrar(imagen, imagen_modificada, "Imagen con Histograma Modificado")
    mostrarHistogramaOriginal(imagen)
    mostrarHistogramaUmbralizada(imagen_modificada, "Histograma Modificado")

    return imagen_modificada
#------------------------------------------------------


#--------- EJECUCIÓN ------------------------------

ruta_original = "lisa.jpg" 
imgLisa = abrir(ruta_original) #Imagen a Color
imgLisaBN = cv2.cvtColor(imgLisa, cv2.COLOR_BGR2GRAY)#imagen a BN




plt.imshow(cv2.cvtColor(imgLisa, cv2.COLOR_BGR2RGB))
plt.title("Imagen Original")
plt.axis('off') 
plt.show()
mostrarHistogramaOriginal(imgLisa)
# Binarización para imagen a color y BN--------------------------------------------------------

umbral = 127 
# Binarización Color
imgLisa_bin_color = Binarizado(imgLisa, umbral)

# Binarización BN
imgLisa_bin_bn = Binarizado(imgLisaBN, umbral)

# Umbralización N niveles-----------------------------------------------------------
niveles = 4
umbrales = [50, 100, 150]  # Tres umbrales para cuatro niveles
valores = [0, 85, 170, 255]  # Los valores a asignar a los píxeles

img_umbr_color = umbralizacion_nivelada(imgLisa, niveles, umbrales, valores)
img_umbr_gris = umbralizacion_nivelada(imgLisaBN, niveles, umbrales, valores)

# Modificación de histograma ---------------------------------------------

n = 50  # Valor mínimo deseado
N = 160 # Valor máximo deseado

imgLisa_modificada = modificar_histograma(imgLisa, n, N)
n = 0  # Valor mínimo deseado
N = 6 # Valor máximo deseado

imgLisa_modificada = modificar_histograma(imgLisaBN, n, N)

