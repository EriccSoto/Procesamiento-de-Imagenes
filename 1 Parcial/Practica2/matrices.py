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
        
   
def cambiarpixel(imagen):
    imagen_modificada = imagen.copy()
    m, n, _ = imagen_modificada.shape #Obtiene medidas de la imagen, particularmente alto y ancho
    color_claro = [random.randint(100, 255) for _ in range(3)]# La salida podría ser algo como: [200, 150, 120]
    imagen_modificada[m//2, n//2, :] = color_claro #color_claro es un arreglo


    cv2.imshow("Imagen Original", imagen)
    cv2.imshow("Imagen con pixel diferente", imagen_modificada)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dibujar_lineas(imagen):
    imagen_modificada = imagen.copy()
    alto, ancho, _ = imagen.shape

    # Coordenadas para las líneas (puntos inicial y final)
    punto_A = (0, 0)  # Punto inicial
    punto_B = (ancho, alto)  # Punto final
    # Dibujar una línea de color azul marino
    cv2.line(imagen_modificada, punto_A, punto_B, (128, 0, 0), 3)  # Azul marino en BGR (128, 0, 0)


    punto_C = (ancho,0)  # Punto inicial 
    punto_D = (0,alto)  # Punto final 
    # Dibujar una línea roja
    cv2.line(imagen_modificada, punto_C, punto_D, (0, 0, 255), 3)  # Rojo en BGR (0, 0, 255), el 3 solo es el grosor de la linea en pixeles

    cv2.imshow("Imagen Original", imagen)
    cv2.imshow("Imagen con 2 lineas", imagen_modificada)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def traslacion(imagen, Tx, Ty):
    
# Obtener las dimensiones de la imagen original
    alto, ancho, canales = imagen.shape

    # Crear una nueva imagen modificada con el tamaño ajustado
    # Nueva altura y ancho considerando los desplazamientos
    nuevo_alto = alto + abs(Ty)
    nuevo_ancho = ancho + abs(Tx)

    # Crear una matriz de ceros con el tamaño ajustado
    imagen_modificada = np.zeros((nuevo_alto, nuevo_ancho, canales), dtype=np.uint8)
    for i in range(imagen_modificada.shape[0]):  # Recorrer las filas (altura)
        for j in range(imagen_modificada.shape[1]):  # Recorrer las columnas (ancho)
            # Calcular las nuevas posiciones con la traslación
            nuevo_i = i + Ty  # Traslación en el eje Y (alto)
            nuevo_j = j + Tx  # Traslación en el eje X (ancho)
            
            # Verificar que las nuevas coordenadas estén dentro de los límites de la imagen
            if 0 <= nuevo_i < imagen_modificada.shape[0] and 0 <= nuevo_j < imagen_modificada.shape[1]:
                # Asignar el valor del píxel de la posición original a la nueva posición
                imagen_modificada[nuevo_i, nuevo_j] = imagen[i, j]

    # Mostrar la imagen traslada
    cv2.imshow("Imagen Original", imagen)
    cv2.imshow("Imagen Trasladada", imagen_modificada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def escalado(imagen, S):
    # Obtener las dimensiones de la imagen original
    alto, ancho, canales = imagen.shape

    # Calcular las nuevas dimensiones
    nuevo_alto = int(alto * S)
    nuevo_ancho = int(ancho * S)

    # Crear una nueva imagen modificada con las nuevas dimensiones
    imagen_modificada = np.zeros((nuevo_alto, nuevo_ancho, canales), dtype=np.uint8)

    # Recorrer cada píxel de la imagen modificada y asignarle un valor de la imagen original
    for i in range(nuevo_alto):  # Recorrer filas (altura)
        for j in range(nuevo_ancho):  # Recorrer columnas (ancho)
            # Calcular las coordenadas correspondientes en la imagen original
            original_i = int(i / S)  # Escalar en el eje Y
            #original_j = int(j / S)  # Escalar en el eje X
            original_j = j//S

            imagen_modificada[i, j] = imagen[original_i, original_j]

    # Mostrar la imagen escalada
    caracteristicas(imagen)
    caracteristicas(imagen_modificada)
    cv2.imshow("Imagen Original", imagen)
    cv2.imshow("Imagen Escalada", imagen_modificada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotacion(imagen, angulo):
    # Convertir el ángulo de grados a radianes
    theta = np.radians(angulo)
    
    # Obtener el tamaño original de la imagen
    alto, ancho = imagen.shape[:2]
    
    # Calcular el nuevo tamaño máximo que puede ocupar la imagen rotada
    nuevo_ancho = int(ancho * abs(np.cos(theta)) + alto * abs(np.sin(theta)))
    nuevo_alto = int(alto * abs(np.cos(theta)) + ancho * abs(np.sin(theta)))
    
    # Crear una imagen con el nuevo tamaño con fondo negro
    imagen_modificada = np.zeros((nuevo_alto, nuevo_ancho, 3), dtype=np.uint8)
    
    # Calcular el nuevo centro de la imagen
    nuevo_cx, nuevo_cy = nuevo_ancho // 2, nuevo_alto // 2
    cx, cy = ancho // 2, alto // 2
    
    # Aplicar la transformación de rotación a cada píxel
    for i in range(alto):
        for j in range(ancho):
            # Trasladar al origen
            x = j - cx
            y = i - cy
            
            # Aplicar las ecuaciones de rotación
            x_rot = int(x * np.cos(theta) - y * np.sin(theta) + nuevo_cx)
            y_rot = int(y * np.cos(theta) + x * np.sin(theta) + nuevo_cy)
            
            # Verificar que los nuevos índices estén dentro de los límites de la imagen ampliada
            if 0 <= x_rot < nuevo_ancho and 0 <= y_rot < nuevo_alto:
                imagen_modificada[y_rot, x_rot] = imagen[i, j]
    
    # Mostrar características de las imágenes
    caracteristicas(imagen)
    caracteristicas(imagen_modificada)
    
    # Mostrar la imagen rotada
    cv2.imshow("Imagen Rotada", imagen_modificada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def rotacion_Pivot(imagen, angulo, xR=None, yR=None):
    
    # Convertir ángulo a radianes
    theta = np.radians(angulo)
    
    # Dimensiones originales de la imagen
    alto, ancho = imagen.shape[:2]
    
    # Si no se especifica un punto pivote, usar el centro de la imagen
    if xR is None or yR is None:
        xR, yR = ancho // 2, alto // 2
    
    # Calcular nuevo tamaño para evitar recortes
    nuevo_ancho = int(ancho * abs(np.cos(theta)) + alto * abs(np.sin(theta)))
    nuevo_alto = int(alto * abs(np.cos(theta)) + ancho * abs(np.sin(theta)))
    
    # Crear una imagen vacía con fondo negro
    imagen_modificada = np.zeros((nuevo_alto, nuevo_ancho, 3), dtype=np.uint8)
    
    # Nuevo centro de la imagen ampliada
    nuevo_cx, nuevo_cy = nuevo_ancho // 2, nuevo_alto // 2
    
    # Aplicar transformación a cada píxel
    for i in range(alto):
        for j in range(ancho):
            # Trasladar al origen usando el punto pivote
            x = j - xR
            y = i - yR
            
            # Aplicar ecuaciones de rotación con punto pivote
            x_rot = int(xR + (x * np.cos(theta) - y * np.sin(theta)))
            y_rot = int(yR + (y * np.cos(theta) + x * np.sin(theta)))
            
            # Ajustar coordenadas a la nueva imagen
            x_rot += nuevo_cx - xR
            y_rot += nuevo_cy - yR
            
            # Verificar si está dentro de los límites
            if 0 <= x_rot < nuevo_ancho and 0 <= y_rot < nuevo_alto:
                imagen_modificada[y_rot, x_rot] = imagen[i, j]
    
    # Mostrar características
    caracteristicas(imagen)
    caracteristicas(imagen_modificada)
    
    # Mostrar imagen rotada
    cv2.imshow("Imagen Rotada", imagen_modificada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def reflexion_x(imagen):
    alto, ancho = imagen.shape[:2]
    imagen_reflejada = np.zeros_like(imagen)

    for i in range(alto):
        for j in range(ancho):
            imagen_reflejada[alto - 1 - i, j] = imagen[i, j]  # Reflejo sobre el eje X

    # Mostrar imagen rotada
    cv2.imshow("Imagen Original", imagen)
    cv2.imshow("Imagen reflejada x", imagen_reflejada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def reflexion_y(imagen):
    
    alto, ancho = imagen.shape[:2]
    imagen_reflejada = np.zeros_like(imagen)

    for i in range(alto):
        for j in range(ancho):
            imagen_reflejada[i, ancho - 1 - j] = imagen[i, j]  # Reflejo sobre el eje Y

    # Mostrar imagen rotada
    cv2.imshow("Imagen Original", imagen)
    cv2.imshow("Imagen reflejada y", imagen_reflejada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def reflexion_xy(imagen):
    
    alto, ancho = imagen.shape[:2]
    imagen_reflejada = np.zeros_like(imagen)

    for i in range(alto):
        for j in range(ancho):
            imagen_reflejada[alto - 1 - i, ancho - 1 - j] = imagen[i, j]  # Reflejo sobre ambos ejes

    # Mostrar imagen rotada
    cv2.imshow("Imagen Original", imagen)
    cv2.imshow("Imagen reflejada xy", imagen_reflejada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







ruta_original = "Imagenes/skz.jpg"
img_skz = abrir(ruta_original)
img_skz_BN = cv2.cvtColor(img_skz, cv2.COLOR_BGR2GRAY)
imagen_creada =crearImg()
#cambiarpixel(imagen_creada)
#dibujar_lineas(img_skz)
#traslacion(img_skz, 80, 30)
#escalado(img_skz, 5)
rotacion(img_skz, 80)
#rotacion_Pivot(img_skz, -30, xR=100, yR=10)
#reflexion_x(img_skz)
#reflexion_y(img_skz)
#reflexion_xy(img_skz)