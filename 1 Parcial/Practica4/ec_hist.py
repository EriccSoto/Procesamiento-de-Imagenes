import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def crearImg():
    return np.zeros((200, 200, 3), dtype=np.uint8)

# Abrir imagen
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

# Mostrar características de la imagen
def caracteristicas(imagen):
    print("\nTipo de imagen:", type(imagen))
    print("Dimensiones (alto, ancho, canales):", imagen.shape)

# Mostrar imágenes originales y modificadas
def mostrar(imagen_original, imagen_modificada, leyenda):
    print("Características de la imagen original:")
    caracteristicas(imagen_original)
    print("Características de la imagen modificada:")
    caracteristicas(imagen_modificada)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Imagen Original
    if (len(imagen_original.shape) == 2):
        axes[0].imshow(imagen_original, cmap="gray")
    else:
        axes[0].imshow(cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB))

    # Imagen Modificada
    if (len(imagen_modificada.shape) == 2):
        axes[1].imshow(imagen_modificada, cmap="gray")
    else:
        axes[1].imshow(cv2.cvtColor(imagen_modificada, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")
    axes[1].set_title(leyenda)
    axes[1].axis("off")

    plt.show()

# Función para calcular el histograma manualmente
def calcular_histograma(imagen, canal=None):
    if len(imagen.shape) == 2:  
        histograma = np.zeros(256)
        for valor in imagen.ravel():  # Convierte la imagen 2D en un arreglo 1D
            histograma[valor] += 1  # Contar las frecuencias de los valores de los píxeles
    else:  # Imagen a color
        if canal is not None:  # Si se especifica un canal
            histograma = np.zeros(256)
            for valor in imagen[:, :, canal].ravel():
                histograma[valor] += 1
        else:  # Si no se especifica un canal
            histograma = np.zeros((256, 3))  # Para tres canales (BGR)
            for i in range(3):
                for valor in imagen[:, :, i].ravel():
                    histograma[valor, i] += 1
                    
    return histograma

# Mostrar el histograma de la imagen original
def mostrarHistogramaOriginal(imagen):
    plt.figure(figsize=(10, 6))

    # Imagen en escala de grises
    if len(imagen.shape) == 2:
        histograma = calcular_histograma(imagen)
        plt.plot(histograma, color="black", label="Escala de Grises")
    else:  # Imagen a color
        colores = ('b', 'g', 'r')  # Blue, Green, Red
        for i, c in enumerate(colores):
            histograma = calcular_histograma(imagen, canal=i)
            plt.plot(histograma, color=c, label=f'Canal {c.upper()}')

    plt.title("Histograma de la Imagen Original")
    plt.xlabel("Intensidad de píxel")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.show()

# Mostrar los histogramas de la imagen modificada
def mostrarHistogramaUmbralizada(imagen, leyenda):
    if imagen is None:
        print("Error: La imagen no se ha cargado correctamente.")
        return

    # Imagen en escala de grises
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
        colores = ('b', 'g', 'r')  # Blue, Green, Red
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, c in enumerate(colores):
            histograma = calcular_histograma(imagen, canal=i)
            axes[i].plot(histograma, color=c)
            axes[i].set_title(f"Histograma Canal {c.upper()}")
            axes[i].set_xlabel("Intensidad de píxel")
            axes[i].set_ylabel("Frecuencia")

        plt.tight_layout()
        plt.show()

# Función para binarizar la imagen
def Binarizado(imagen, umbral):
    if len(imagen.shape) == 2:  # Imagen en escala de grises
        imagen_binarizada = np.where(imagen >= umbral, 255, 0).astype(np.uint8)
    else:  # Imagen a color
        imagen_binarizada = np.zeros_like(imagen)
        imagen_binarizada[:, :, :] = np.where(imagen[:, :, :] >= umbral, 255, 0)

    mostrar(imagen, imagen_binarizada, "Imagen Binarizada")
    mostrarHistogramaOriginal(imagen)
    mostrarHistogramaUmbralizada(imagen_binarizada, "Imagen Binarizada")

    return imagen_binarizada

# Función de umbralización con N niveles
def umbralizacion_nivelada(imagen, niveles, umbrales, valores):
    """
    Realiza una umbralización de la imagen en base a N niveles.
    
    :param imagen: Imagen de entrada (puede ser en escala de grises o a color).
    :param niveles: Número de niveles de umbralización.
    :param umbrales: Lista con los valores de los umbrales (tamaño N-1).
    :param valores: Lista con los valores a asignar a los píxeles en cada intervalo (tamaño N).
    
    :return: Imagen umbralizada.
    """
    # Verificar si la imagen es en escala de grises o a color
    if len(imagen.shape) == 2:  # Imagen en escala de grises
        imagen_umbr = np.zeros_like(imagen)
        for x in range(imagen.shape[0]):
            for y in range(imagen.shape[1]):
                valor = imagen[x, y]
                for i in range(niveles - 1):
                    if umbrales[i] < valor <= umbrales[i + 1]:
                        imagen_umbr[x, y] = valores[i + 1]  # Asignar el valor correspondiente
                        break
                    elif valor <= umbrales[0]:
                        imagen_umbr[x, y] = valores[0]
                        break
                    elif valor > umbrales[-1]:
                        imagen_umbr[x, y] = valores[-1]
                        break
    else:  # Imagen a color
        imagen_umbr = np.zeros_like(imagen)
        for i in range(3):  # Procesar los tres canales (B, G, R)
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







# ---------- EJECUCIÓN ----------  
ruta_original = "lisa.jpg"  # Cambia esta ruta por la de tu imagen  
umbral = 127  # Puedes cambiar el umbral  

# Abrir la imagen original
imgLisa = abrir(ruta_original)
"""""
# Binarización para imagen a color
imgLisa_bin_color = Binarizado(imgLisa, umbral)

# Convertir la imagen a escala de grises y binarizarla
imgLisaBN = cv2.cvtColor(imgLisa, cv2.COLOR_BGR2GRAY)
imgLisa_bin_bn = Binarizado(imgLisaBN, umbral)

# Umbralización nivelada para imagen a color
niveles = 4
umbrales = [50, 100, 150]  # Tres umbrales para cuatro niveles
valores = [0, 85, 170, 255]  # Los valores a asignar a los píxeles

img_umbr_color = umbralizacion_nivelada(imgLisa, niveles, umbrales, valores)
img_umbr_gris = umbralizacion_nivelada(imgLisaBN, niveles, umbrales, valores)

"""""










import numpy as np

def modificar_histograma(imagen, n, N):
    """
    Modifica el histograma de una imagen usando la fórmula.
    
    :param imagen: Imagen de entrada 
    :param n: Valor mínimo deseado en la imagen modificada.
    :param N: Valor máximo deseado en la imagen modificada.
    
    :return: Imagen con el histograma modificado.
    """
    # Verificar si la imagen es en escala de grises o a color
    if len(imagen.shape) == 2:  # Imagen en escala de grises
        m = np.min(imagen)  # Valor mínimo en la imagen original
        M = np.max(imagen)  # Valor máximo en la imagen original
        
        # Convertir la imagen a float32 para evitar desbordamientos
        imagen = imagen.astype(np.float32)
        
        # Aplicar la fórmula de modificación del histograma
        imagen_modificada = n + ((imagen - m) * (N - n)) / (M - m)
        
    else:  # Imagen a color
        imagen_modificada = np.zeros_like(imagen, dtype=np.float32)  # Usar float32 para evitar desbordamientos
        for i in range(3):  # Procesar los tres canales (B, G, R)
            canal = imagen[:, :, i].astype(np.float32)  # Convertir cada canal a float32
            m = np.min(canal)  # Valor mínimo en el canal
            M = np.max(canal)  # Valor máximo en el canal
            
            # Aplicar la fórmula de modificación del histograma al canal
            imagen_modificada[:, :, i] = n + ((canal - m) * (N - n)) / (M - m)
    
    # Convertir la imagen modificada a uint8
    imagen_modificada = imagen_modificada.astype(np.uint8)  
    
    # Mostrar la imagen original y la modificada
    mostrar(imagen, imagen_modificada, "Imagen con Histograma Modificado")
    mostrarHistogramaOriginal(imagen)
    mostrarHistogramaUmbralizada(imagen_modificada, "Histograma Modificado")

    return imagen_modificada


# ---------- EJECUCIÓN ----------  
ruta_original = "lisa.jpg"  # Cambia esta ruta por la de tu imagen  
umbral = 127  # Puedes cambiar el umbral  

# Abrir la imagen original
imgLisa = abrir(ruta_original)

n = 50  # Valor mínimo deseado
N = 120 # Valor máximo deseado

imgLisa_modificada = modificar_histograma(imgLisa, n, N)
