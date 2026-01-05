import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para cargar la imagen
def cargar_imagen(ruta):
    imagen = cv2.imread(ruta, cv2.IMREAD_COLOR)  # Lee la imagen en color

    # Muestra la imagen
    cv2.imshow('Imagen Original', imagen)

    # Espera hasta que se presione una tecla
    cv2.waitKey(0)

    # Cierra todas las ventanas de OpenCV
    cv2.destroyAllWindows()
    return cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

# Función para mostrar la imagen
def mostrar_imagen(titulo, imagen):
    imagen = imagen.astype(np.uint8)  # Convierte a tipo uint8
    cv2.imshow(titulo, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 1. Filtro Gaussiano
def filtro_gaussiano(imagen):
    mascara_gaussiana = np.array([
        [1, 1, 2, 2, 2, 1, 1],
        [1, 2, 2, 4, 2, 2, 1],
        [2, 2, 4, 8, 4, 2, 2],
        [2, 4, 8, 16, 8, 4, 2],
        [2, 2, 4, 8, 4, 2, 2],
        [1, 2, 2, 4, 2, 2, 1],
        [1, 1, 2, 2, 2, 1, 1]
    ])
    return aplicar_mascara(imagen, mascara_gaussiana)

# 2. Máscara de Prewitt Norte (Detecta bordes horizontales)
def mascara_prewitt_Norte(imagen):
    mascara = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    return aplicar_mascara(imagen, mascara)

# 3. Máscara de Prewitt Oeste (Detecta bordes verticales)
def mascara_prewitt_Oeste(imagen):
    mascara = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ])
    return aplicar_mascara(imagen, mascara)

# 4. Máscara de Prewitt Este
def mascara_prewitt_Este(imagen):
    mascara = np.array([
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1]
    ])
    return aplicar_mascara(imagen, mascara)

# 5. Máscara de Prewitt Sur
def mascara_prewitt_Sur(imagen):
    mascara = np.array([
        [1,   2,  1],
        [0,   0,  0],
        [-1,  -2,  -1]
    ])
    return aplicar_mascara(imagen, mascara)

# 6. Máscara de Prewitt Noreste
def mascara_prewitt_Noreste(imagen):
    mascara = np.array([
        [0,  -1,  -2],
        [1,  0,  -1],
        [2,  1,  0]
    ])
    return aplicar_mascara(imagen, mascara)

# 7. Máscara de Prewitt Sureste
def mascara_prewitt_Sureste(imagen):
    mascara = np.array([
        [ 2,  1,  0],
        [ 1,  0,  -1],
        [ 0, -1, -2]
    ])
    return aplicar_mascara(imagen, mascara)

# 8. Máscara de Prewitt Suroeste
def mascara_prewitt_Suroeste(imagen):
    mascara = np.array([
        [0, 1, 2],
        [ -1,  0,  1],
        [ -2,  -1,  0]
    ])
    return aplicar_mascara(imagen, mascara)

# 9. Máscara de Prewitt Noroeste
def mascara_prewitt_Noroeste(imagen):
    mascara = np.array([
        [ -2,  -1,  0],
        [ -1,  0,  1],
        [0, 1, 2]
    ])
    return aplicar_mascara(imagen, mascara)

# 12. Máscara pasa altas 3x3
def mascara_pasa_altas_3x3(imagen):
    mascara = np.array([
        [ -1, -1, -1],
        [-1,  9, -1],
        [-1, -1,  -1]
    ])
    return aplicar_mascara(imagen, mascara)
#------------------------------------------------------------------------------------------
# 13. Máscara pasa altas 5x5
def mascara_pasa_altas_5x5(imagen):
    mascara = np.array([
        [ 0, -1, -1, -1,  0],
        [-1,  2, -4,  2, -1],
        [-1, -4, 13, -4, -1],
        [-1,  2, -4,  2, -1],
        [ 0, -1, -1, -1,  0]
    ])
    return aplicar_mascara(imagen, mascara)

# 14. Máscara pasa bajas 3x3
def mascara_pasa_bajas_3x3(imagen):
    mascara = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 9
    return aplicar_mascara(imagen, mascara)

# 15. Máscara pasa bajas 5x5
def mascara_pasa_bajas_5x5(imagen):
    mascara = np.array([
        [1/25, 1/25, 1/25, 1/25, 1/25],
        [1/25, 4/25, 4/25, 4/25, 1/25],
        [1/25, 4/25, 12/25, 4/25, 1/25],
        [1/25, 4/25, 4/25, 4/25, 1/25],
        [1/25, 1/25, 1/25, 1/25, 1/25]
    ]) 
    return aplicar_mascara(imagen, mascara)

# 16. Laplaciano 3x3
def laplaciano_3x3(imagen):
    mascara = np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ])
    return aplicar_mascara(imagen, mascara)

# 17. Laplaciano 5x5
def laplaciano_5x5(imagen):
    mascara = np.array([
        [ -1,  -1, -1,  -1,  -1],
        [ -1, -1, -1, -1,  -1],
        [-1, -1, 24, -1, -1],
        [ -1, -1, -1, -1,  -1],
        [ -1,  -1, -1,  -1,  -1]
    ])
    return aplicar_mascara(imagen, mascara)
#------------------------------------------------------------------------------------------
# 18. Filtro de Robert 
def filtro_robert(imagen):
    mascara = np.array([
        [ 0, -1],
        [ -1,  0]
    ])
    return aplicar_mascara(imagen, mascara)

# 20. Filtro Sobel Horizontal
def filtro_sobel_horizontal(imagen):
    mascara = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    return aplicar_mascara(imagen, mascara)

# 21. Filtro Sobel Vertical
def filtro_sobel_vertical(imagen):
    mascara = np.array([
        [-1,  0,  -1],
        [-2,  0,  -2],
        [-1,  0,  -1]
    ])
    return aplicar_mascara(imagen, mascara)

# 24. Filtro Prewitt Horizontal
def filtro_prewitt_horizontal(imagen):
    mascara = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ])
    return aplicar_mascara(imagen, mascara)

# 25. Filtro Prewitt Vertical
def filtro_prewitt_vertical(imagen):
    mascara = np.array([
        [-1,  0,  1],
        [-1,  0,  1],
        [-1,  0,  1]
    ])
    return aplicar_mascara(imagen, mascara)

# 26. Operador Isotrópico (Filtro Isotrópico)
def operador_isotropico1(imagen):
    mascara = np.array([
        [-1, 0, 1],
        [-np.sqrt(2), 0, np.sqrt(2)],
        [-1, 0, 1]
    ])
    return aplicar_mascara(imagen, mascara)

def operador_isotropico2(imagen):
    mascara = np.array([
        [-1, -np.sqrt(2), -1],
        [0, 0, 0],
        [1, np.sqrt(2), 1]
    ])
    return aplicar_mascara(imagen, mascara)

# Filtro Promedio
def filtro_promedio(imagen):
    mascara = np.array([
        [1/3, 1/3, 1/3],
        [1/3, 1/3, 1/3],
        [1/3, 1/3, 1/3]
    ])
    return aplicar_mascara(imagen, mascara)

# 28. Filtro de Media Ponderada
def filtro_media_ponderada(imagen):
    mascara = np.array([
        [1/16, 1/16, 1/16],
        [1/16, 8/16, 1/16],
        [1/16, 1/16, 1/16]
    ]) 
    return aplicar_mascara(imagen, mascara)

def laplaciano_2x2_1(imagen):
    mascara = np.array([
        [ 0, 0],
        [-1,  1]
    ])
    return aplicar_mascara(imagen, mascara)

# 30. Operador Discreto del Laplaciano 2x2 (máscara 2)
def laplaciano_2x2_2(imagen):
    mascara = np.array([
        [ 1, 0],
        [-1,  0]
    ])
    return aplicar_mascara(imagen, mascara)

def laplaciano_2x2_3(imagen):
    mascara = np.array([
        [ 1, 0],
        [0,  -1]
    ])
    return aplicar_mascara(imagen, mascara)

def laplaciano_2x2_4(imagen):
    mascara = np.array([
        [ 0, 1],
        [-1,  0]
    ])
    return aplicar_mascara(imagen, mascara)

# 31. Operador Discreto del Laplaciano 3x3 (máscara 1)
def laplaciano_3x3_1(imagen):
    mascara = np.array([
        [ 1, 1,  1],
        [0,  0, 0],
        [ -1, -1,  -1]
    ])
    return aplicar_mascara(imagen, mascara)

# 32. Operador Discreto del Laplaciano 3x3 (máscara 2)
def laplaciano_3x3_2(imagen):
    mascara = np.array([
        [1, 0, -1],
        [1,  0, -1],
        [1, 0, -1]
    ])
    return aplicar_mascara(imagen, mascara)

# 33. Operador Discreto del Laplaciano 3x3 (máscara 3)
def laplaciano_3x3_3(imagen):
    mascara = np.array([
        [ 1,  2,  1],
        [ 0, 0,  0],
        [ -1,  -2,  -1]
    ])
    return aplicar_mascara(imagen, mascara)

# 34. Operador Discreto del Laplaciano 3x3 (máscara 4)
def laplaciano_3x3_4(imagen):
    mascara = np.array([
        [ 1,  0, -1],
        [ 2, 0,  -2],
        [ 1,  0, -1]
    ])
    return aplicar_mascara(imagen, mascara)

# 34. Operador Discreto del Laplaciano 3x3 (máscara 5)
def laplaciano_3x3_5(imagen):
    mascara = np.array([
        [ 0,  1, 0],
        [ 1, -4,  1],
        [ 1,  1, 0]
    ])
    return aplicar_mascara(imagen, mascara)

# 34. Operador Discreto del Laplaciano 3x3 (máscara 6)
def laplaciano_3x3_6(imagen):
    mascara = np.array([
        [ 1,  1, 1],
        [ 1, -8,  1],
        [ 1,  1, 1]
    ])
    return aplicar_mascara(imagen, mascara)

# 34. Operador Discreto del Laplaciano 3x3 (máscara 7)
def laplaciano_3x3_7(imagen):
    mascara = np.array([
        [ -1,  2, -1],
        [ 2, -4,  2],
        [ -1,  2, -1]
    ])
    return aplicar_mascara(imagen, mascara)

# 35. Máscara de Enfoque
def mascara_enfoque(imagen):
    mascara = np.array([
        [ 1/16, -4/16,  1/16],
        [-4/16, 26/16, -4/16],
        [ 1/16, -4/16,  1/16]
    ])
    return aplicar_mascara(imagen, mascara)

# 36. Máscaras Artísticas (1)
def mascara_artistica_1(imagen):
    mascara = np.array([
        [-2*95, -1*95,  0*95],
        [-1*95,  1*95,  1*95],
        [ 0*95,  1*95,  2*95]
    ])

    mascara2 = np.array([
        [-1, -4, -1],
        [-4, 26, -4],
        [-1, -4, -1]
    ])
    mascara3 =mascara + mascara2
    return aplicar_mascara(imagen, mascara3)

def aplicar_mascara(imagen, mascara):
    M, N = imagen.shape      # D imagen
    m, n = mascara.shape   #  máscara
    print(f"Dimensiones de la imagen: {M}x{N}")
    print(f"Dimensiones de la máscara: {m}x{n}")
    resultado = np.zeros_like(imagen)  # matriz de ceros
    print(f"M - m + 1---{M} - {m} + {1}= {M - m + 1}")
    print(f"N - n + 1---{N} - {n} + {1}= {N - n + 1}")
    
    for i in range(M - m + 1): 
        for j in range(N - n + 1): 
            suma = 0 

           
            for k in range(m): 
                for l in range(n):  
                    suma += imagen[i + k, j + l] * mascara[k, l]

            resultado[i+1, j+1] = suma

    return resultado


if __name__ == "__main__":
    ruta_imagen = 'bici.jpg'  
    imagen = cargar_imagen(ruta_imagen)

    
    
    # 1. Filtro Gaussiano
    imagen_gaussiana = filtro_gaussiano(imagen)
    mostrar_imagen("Imagen con Filtro Gaussiano", imagen_gaussiana)
    
    # 2. Máscara de Prewitt Norte
    imagen_prewitt_norte = mascara_prewitt_Norte(imagen)
    mostrar_imagen("Bordes con Prewitt Norte", imagen_prewitt_norte)
    
   
    # 3. Máscara de Prewitt Oeste
    imagen_prewitt_oeste = mascara_prewitt_Oeste(imagen)
    mostrar_imagen("Bordes con Prewitt Oeste", imagen_prewitt_oeste)
    
    # 4. Máscara de Prewitt Este
    imagen_prewitt_este = mascara_prewitt_Este(imagen)
    mostrar_imagen("Bordes con Prewitt Este", imagen_prewitt_este)
    
    # 5. Máscara de Prewitt Sur
    imagen_prewitt_sur = mascara_prewitt_Sur(imagen)
    mostrar_imagen("Bordes con Prewitt Sur", imagen_prewitt_sur)
    
    # 6. Máscara de Prewitt Noreste
    imagen_prewitt_noreste = mascara_prewitt_Noreste(imagen)
    mostrar_imagen("Bordes con Prewitt Noreste", imagen_prewitt_noreste)
    
    # 7. Máscara de Prewitt Sureste
    imagen_prewitt_sureste = mascara_prewitt_Sureste(imagen)
    mostrar_imagen("Bordes con Prewitt Sureste", imagen_prewitt_sureste)
    
    # 8. Máscara de Prewitt Suroeste
    imagen_prewitt_suroeste = mascara_prewitt_Suroeste(imagen)
    mostrar_imagen("Bordes con Prewitt Suroeste", imagen_prewitt_suroeste)
    
    # 9. Máscara de Prewitt Noroeste
    imagen_prewitt_noroeste = mascara_prewitt_Noroeste(imagen)
    mostrar_imagen("Bordes con Prewitt Noroeste", imagen_prewitt_noroeste)
    
    # 12. Máscara pasa altas 3x3
    imagen_pasa_altas_3x3 = mascara_pasa_altas_3x3(imagen)
    mostrar_imagen("Imagen con Pasa Altas 3x3", imagen_pasa_altas_3x3)
    
    # 13. Máscara pasa altas 5x5
    imagen_pasa_altas_5x5 = mascara_pasa_altas_5x5(imagen)
    mostrar_imagen("Imagen con Pasa Altas 5x5", imagen_pasa_altas_5x5)
    
    # 14. Máscara pasa bajas 3x3
    imagen_pasa_bajas_3x3 = mascara_pasa_bajas_3x3(imagen)
    mostrar_imagen("Imagen con Pasa Bajas 3x3", imagen_pasa_bajas_3x3)
    
    # 15. Máscara pasa bajas 5x5
    imagen_pasa_bajas_5x5 = mascara_pasa_bajas_5x5(imagen)
    mostrar_imagen("Imagen con Pasa Bajas 5x5", imagen_pasa_bajas_5x5)
    
    # 16. Laplaciano 3x3
    imagen_laplaciano_3x3 = laplaciano_3x3(imagen)
    mostrar_imagen("Imagen con Laplaciano 3x3", imagen_laplaciano_3x3)
    
    # 17. Laplaciano 5x5
    imagen_laplaciano_5x5 = laplaciano_5x5(imagen)
    mostrar_imagen("Imagen con Laplaciano 5x5", imagen_laplaciano_5x5)
    
    # 18. Filtro de Robert
    imagen_filtro_robert = filtro_robert(imagen)
    mostrar_imagen("Imagen con Filtro de Robert", imagen_filtro_robert)
    
    # 20. Filtro Sobel Horizontal
    imagen_sobel_horizontal = filtro_sobel_horizontal(imagen)
    mostrar_imagen("Imagen con Sobel Horizontal", imagen_sobel_horizontal)
    
    # 21. Filtro Sobel Vertical
    imagen_sobel_vertical = filtro_sobel_vertical(imagen)
    mostrar_imagen("Imagen con Sobel Vertical", imagen_sobel_vertical)
    
    # 24. Filtro Prewitt Horizontal
    imagen_prewitt_horizontal = filtro_prewitt_horizontal(imagen)
    mostrar_imagen("Imagen con Prewitt Horizontal", imagen_prewitt_horizontal)
    
    # 25. Filtro Prewitt Vertical
    imagen_prewitt_vertical = filtro_prewitt_vertical(imagen)
    mostrar_imagen("Imagen con Prewitt Vertical", imagen_prewitt_vertical)
    
    # 26. Operador Isotrópico 1
    imagen_isotropico1 = operador_isotropico1(imagen)
    mostrar_imagen("Imagen con Operador Isotrópico 1", imagen_isotropico1)
    
    # 27. Operador Isotrópico 2
    imagen_isotropico2 = operador_isotropico2(imagen)
    mostrar_imagen("Imagen con Operador Isotrópico 2", imagen_isotropico2)
    
    # 28. Filtro Promedio
    imagen_promedio = filtro_promedio(imagen)
    mostrar_imagen("Imagen con Filtro Promedio", imagen_promedio)
    
    # 29. Filtro Media Ponderada
    imagen_media_ponderada = filtro_media_ponderada(imagen)
    mostrar_imagen("Imagen con Filtro Media Ponderada", imagen_media_ponderada)
    
    # 30. Laplaciano 2x2 1
    imagen_laplaciano_2x2_1 = laplaciano_2x2_1(imagen)
    mostrar_imagen("Imagen con Laplaciano 2x2 1", imagen_laplaciano_2x2_1)
    
    # 31. Laplaciano 2x2 2
    imagen_laplaciano_2x2_2 = laplaciano_2x2_2(imagen)
    mostrar_imagen("Imagen con Laplaciano 2x2 2", imagen_laplaciano_2x2_2)
    
    # 32. Laplaciano 2x2 3
    imagen_laplaciano_2x2_3 = laplaciano_2x2_3(imagen)
    mostrar_imagen("Imagen con Laplaciano 2x2 3", imagen_laplaciano_2x2_3)
    
    # 33. Laplaciano 2x2 4
    imagen_laplaciano_2x2_4 = laplaciano_2x2_4(imagen)
    mostrar_imagen("Imagen con Laplaciano 2x2 4", imagen_laplaciano_2x2_4)
    
    # 34. Laplaciano 3x3 1
    imagen_laplaciano_3x3_1 = laplaciano_3x3_1(imagen)
    mostrar_imagen("Imagen con Laplaciano 3x3 1", imagen_laplaciano_3x3_1)
    
    # 35. Laplaciano 3x3 2
    imagen_laplaciano_3x3_2 = laplaciano_3x3_2(imagen)
    mostrar_imagen("Imagen con Laplaciano 3x3 2", imagen_laplaciano_3x3_2)
    
    # 36. Laplaciano 3x3 3
    imagen_laplaciano_3x3_3 = laplaciano_3x3_3(imagen)
    mostrar_imagen("Imagen con Laplaciano 3x3 3", imagen_laplaciano_3x3_3)
    
    # 37. Laplaciano 3x3 4
    imagen_laplaciano_3x3_4 = laplaciano_3x3_4(imagen)
    mostrar_imagen("Imagen con Laplaciano 3x3 4", imagen_laplaciano_3x3_4)
    
    # 38. Laplaciano 3x3 5
    imagen_laplaciano_3x3_5 = laplaciano_3x3_5(imagen)
    mostrar_imagen("Imagen con Laplaciano 3x3 5", imagen_laplaciano_3x3_5)
    
    # 39. Laplaciano 3x3 6
    imagen_laplaciano_3x3_6 = laplaciano_3x3_6(imagen)
    mostrar_imagen("Imagen con Laplaciano 3x3 6", imagen_laplaciano_3x3_6)
    
    # 40. Laplaciano 3x3 7
    imagen_laplaciano_3x3_7 = laplaciano_3x3_7(imagen)
    mostrar_imagen("Imagen con Laplaciano 3x3 7", imagen_laplaciano_3x3_7)
    
    # 41. Máscara de Enfoque
    imagen_enfoque = mascara_enfoque(imagen)
    mostrar_imagen("Imagen con Máscara de Enfoque", imagen_enfoque)
    
    # 42. Máscaras Artísticas (1)
    imagen_artistica_1 = mascara_artistica_1(imagen)
    mostrar_imagen("Imagen con Máscara Artística 1", imagen_artistica_1)
