import cv2 
import numpy as np

# ---------- CANNY (usando funciones manuales) ----------
def cargar_imagen_gris(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32)

def cargar_imagen_color(ruta):
    return cv2.imread(ruta, cv2.IMREAD_COLOR)

def aplicar_mascara(imagen, mascara):
    M, N = imagen.shape
    m, n = mascara.shape
    pad_y = m // 2
    pad_x = n // 2
    imagen_padded = np.pad(imagen, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)
    resultado = np.zeros_like(imagen, dtype=np.float32)
    for i in range(M):
        for j in range(N):
            suma = 0
            for k in range(m):
                for l in range(n):
                    suma += imagen_padded[i + k, j + l] * mascara[k, l]
            resultado[i, j] = suma
    return resultado

def filtro_gaussiano(imagen):
    kernel = np.array([
        [2,  4,  5,  4, 2],
        [4,  9, 12,  9, 4],
        [5, 12, 15, 12, 5],
        [4,  9, 12,  9, 4],
        [2,  4,  5,  4, 2]
    ], dtype=np.float32) / 159.0
    return aplicar_mascara(imagen, kernel)

def sobel_gradiente(imagen):
    sobel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]], dtype=np.float32)
    sobel_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]], dtype=np.float32)
    Gx = aplicar_mascara(imagen, sobel_x)
    Gy = aplicar_mascara(imagen, sobel_y)
    altura, ancho = Gx.shape
    magnitud = np.zeros((altura, ancho), dtype=np.float32)
    direccion = np.zeros((altura, ancho), dtype=np.float32)
    for i in range(altura):
        for j in range(ancho):
            magnitud[i][j] = np.sqrt(Gx[i][j]**2 + Gy[i][j]**2)
            direccion[i][j] = np.arctan2(Gy[i][j], Gx[i][j])
    return magnitud, direccion

def supresion_no_maxima(magnitud, direccion):
    altura, ancho = magnitud.shape
    resultado = np.zeros((altura, ancho), dtype=np.float32)
    angulo = direccion * 180 / np.pi
    angulo[angulo < 0] += 180
    for i in range(1, altura - 1):
        for j in range(1, ancho - 1):
            q = r = 255
            a = angulo[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = magnitud[i, j + 1]
                r = magnitud[i, j - 1]
            elif (22.5 <= a < 67.5):
                q = magnitud[i + 1, j - 1]
                r = magnitud[i - 1, j + 1]
            elif (67.5 <= a < 112.5):
                q = magnitud[i + 1, j]
                r = magnitud[i - 1, j]
            elif (112.5 <= a < 157.5):
                q = magnitud[i - 1, j - 1]
                r = magnitud[i + 1, j + 1]

            if (magnitud[i, j] >= q) and (magnitud[i, j] >= r):
                resultado[i, j] = magnitud[i, j]
            else:
                resultado[i, j] = 0
    return resultado

def umbral_doble(imagen, bajo, alto):
    altura, ancho = imagen.shape
    res = np.zeros((altura, ancho), dtype=np.uint8)
    fuerte = 255
    debil = 50
    for i in range(altura):
        for j in range(ancho):
            val = imagen[i, j]
            if val >= alto:
                res[i, j] = fuerte
            elif val >= bajo:
                res[i, j] = debil
            else:
                res[i, j] = 0
    return res, fuerte, debil

def histéresis(imagen, fuerte, debil):
    altura, ancho = imagen.shape
    for i in range(1, altura - 1):
        for j in range(1, ancho - 1):
            if imagen[i, j] == debil:
                vecinos = [
                    imagen[i+1, j-1], imagen[i+1, j], imagen[i+1, j+1],
                    imagen[i, j-1],                 imagen[i, j+1],
                    imagen[i-1, j-1], imagen[i-1, j], imagen[i-1, j+1],
                ]
                if any(vec == fuerte for vec in vecinos):
                    imagen[i, j] = fuerte
                else:
                    imagen[i, j] = 0
    return imagen

def canny_manual(ruta):
    img = cargar_imagen_gris(ruta)
    cv2.imshow("1. Imagen Grayscale", img.astype(np.uint8))

    paso1 = filtro_gaussiano(img)
    cv2.imshow("2. Desenfoque Gaussiano", paso1.astype(np.uint8))

    magnitud, direccion = sobel_gradiente(paso1)
    cv2.imshow("3. Magnitud Gradiente", np.clip(magnitud, 0, 255).astype(np.uint8))

    paso3 = supresion_no_maxima(magnitud, direccion)
    cv2.imshow("4. Supresión No Máxima", np.clip(paso3, 0, 255).astype(np.uint8))

    paso4, fuerte, debil = umbral_doble(paso3, 20, 40)
    cv2.imshow("5. Umbral Doble", paso4)

    paso5 = histéresis(paso4, fuerte, debil)
    cv2.imshow("6. Resultado Histeresis (Canny Final)", paso5)

    return paso5

# ---------- DILATACIÓN ----------
def dilatacion_manual(imagen_binaria, tamaño_kernel=(5,5)):
    alto, ancho = imagen_binaria.shape
    k_alto, k_ancho = tamaño_kernel
    pad_y = k_alto // 2
    pad_x = k_ancho // 2
    padded = np.zeros((alto + 2*pad_y, ancho + 2*pad_x), dtype=np.uint8)
    padded[pad_y:pad_y+alto, pad_x:pad_x+ancho] = imagen_binaria
    salida = np.zeros_like(imagen_binaria)
    for i in range(alto):
        for j in range(ancho):
            ventana = padded[i:i+k_alto, j:j+k_ancho]
            if np.any(ventana == 255):
                salida[i, j] = 255
    return salida

# ---------- RECORTE Y MONTAJE ----------
def recortar_objeto(imagen_color, mascara_binaria):
    h, w, _ = imagen_color.shape
    objeto = np.zeros_like(imagen_color)
    for y in range(h):
        for x in range(w):
            if mascara_binaria[y, x] == 255:
                objeto[y, x] = imagen_color[y, x]
    return objeto

def insertar_sobre_fondo(objeto, fondo):
    h_f, w_f, _ = fondo.shape
    h_o, w_o, _ = objeto.shape
    if h_o > h_f or w_o > w_f:
        raise ValueError("El objeto es más grande que el fondo. No se puede insertar sin redimensionar.")
    offset_y = (h_f - h_o) // 2
    offset_x = (w_f - w_o) // 2
    fondo_copy = fondo.copy()
    for y in range(h_o):
        for x in range(w_o):
            if np.any(objeto[y, x] != 0):
                fondo_copy[offset_y + y, offset_x + x] = objeto[y, x]
    return fondo_copy

# ---------- RELLENAR CONTORNO PRINCIPAL ----------
def rellenar_contorno_principal(imagen_binaria):
    binaria = imagen_binaria.astype(np.uint8)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return binaria
    contorno_principal = max(contornos, key=cv2.contourArea)
    mascara_rellena = np.zeros_like(binaria)
    cv2.drawContours(mascara_rellena, [contorno_principal], -1, 255, thickness=cv2.FILLED)
    return mascara_rellena

# ---------- MAIN ----------
ruta_objeto = "dog.jpg"
ruta_fondo = "dessert.jpg"

# 1. Detectamos bordes con Canny
canny = canny_manual(ruta_objeto)

# 2. Dilatamos para cerrar bordes
dilatada = dilatacion_manual(canny)
cv2.imshow("7. Dilatación", dilatada)

# 3. Rellenamos el contorno principal
mascara_objeto = rellenar_contorno_principal(dilatada)
cv2.imshow("8. Contorno Principal Rellenado", mascara_objeto)

# 4. Recortamos el objeto
imagen_color = cargar_imagen_color(ruta_objeto)
objeto_recortado = recortar_objeto(imagen_color, mascara_objeto)
cv2.imshow("9. Objeto Recortado", objeto_recortado)

# 5. Insertamos sobre fondo
fondo = cargar_imagen_color(ruta_fondo)
resultado = insertar_sobre_fondo(objeto_recortado, fondo)
cv2.imshow("10. Fondo Original", fondo)
cv2.imshow("11. Resultado Final", resultado)

cv2.waitKey(0)
cv2.destroyAllWindows()
