import cv2
import numpy as np

imagen_original = cv2.imread("dog3.jpg")

def convolucion_sobel(imagen, kernel):
    alto, ancho = imagen.shape
    kernel_tam = kernel.shape[0]
    pad_size = kernel_tam // 2
    imagen_padded = np.pad(imagen, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
    convolucion_matriz = np.zeros_like(imagen)

    for i in range(alto):
        for j in range(ancho):
            suma = 0
            for m in range(kernel_tam):
                for n in range(kernel_tam):
                    suma += imagen_padded[i + m, j + n] * kernel[m, n]
            convolucion_matriz[i, j] = suma

    return convolucion_matriz

def detectar_bordes_sobel(imagen):
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, imagen_bn = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("Paso 1 - Imagen binarizada", imagen_bn)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    gradiente_x = convolucion_sobel(imagen_bn, sobel_x)
    gradiente_y = convolucion_sobel(imagen_bn, sobel_y)

    magnitud_gradiente = np.sqrt(gradiente_x.astype(float)**2 + gradiente_y.astype(float)**2)
    magnitud_gradiente = np.clip(magnitud_gradiente, 0, 255).astype(np.uint8)

    cv2.imshow("Paso 2 -Sobel", magnitud_gradiente)

    return magnitud_gradiente

def cierre_morfologico_manual(imagen_binaria, tam_kernel=3):
    alto, ancho = imagen_binaria.shape
    pad = tam_kernel // 2

    # DILATACIÓN
    dilatada = np.zeros_like(imagen_binaria)
    for i in range(pad, alto - pad):
        for j in range(pad, ancho - pad):
            max_val = 0
            for m in range(-pad, pad + 1):
                for n in range(-pad, pad + 1):
                    if imagen_binaria[i + m, j + n] > max_val:
                        max_val = imagen_binaria[i + m, j + n]
            dilatada[i, j] = max_val


    # EROSIÓN
    cerrada = np.zeros_like(imagen_binaria)
    for i in range(pad, alto - pad):
        for j in range(pad, ancho - pad):
            min_val = 255
            for m in range(-pad, pad + 1):
                for n in range(-pad, pad + 1):
                    if dilatada[i + m, j + n] < min_val:
                        min_val = dilatada[i + m, j + n]
            cerrada[i, j] = min_val

    cv2.imshow("Paso 3 - Cierre (Dilatación + Erosión)", cerrada)

    return cerrada

def flood_fill_manual(mask, x, y, marcador=128):
    alto, ancho = mask.shape
    if mask[x, y] != 0:
        return

    pila = [(x, y)]
    while pila:
        i, j = pila.pop()
        if i < 0 or i >= alto or j < 0 or j >= ancho:
            continue
        if mask[i, j] != 0:
            continue
        mask[i, j] = marcador
        pila.append((i+1, j))
        pila.append((i-1, j))
        pila.append((i, j+1))
        pila.append((i, j-1))

def rellenar_segmento_manual(mask):
    relleno = np.copy(mask)

    # Rellenamos desde el fondo (esquinas)
    flood_fill_manual(relleno, 0, 0, marcador=255)
    flood_fill_manual(relleno, 0, relleno.shape[1]-1, marcador=255)
    flood_fill_manual(relleno, relleno.shape[0]-1, 0, marcador=255)
    flood_fill_manual(relleno, relleno.shape[0]-1, relleno.shape[1]-1, marcador=255)

    # Invertimos: lo que NO se llenó es el objeto
    for i in range(relleno.shape[0]):
        for j in range(relleno.shape[1]):
            if relleno[i, j] == 0:
                relleno[i, j] = 255
            else:
                relleno[i, j] = 0

    cv2.imshow("Paso 4 - Máscara segmentada rellenada", relleno)

    return relleno

def superponer_segmentacion(imagen, mascara, color=(0, 255, 0), alpha=0.5):
    imagen_overlay = imagen.copy()
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            if mascara[i, j] > 0:
                for c in range(3):
                    imagen_overlay[i, j, c] = int((1 - alpha) * imagen[i, j, c] + alpha * color[c])
    return imagen_overlay

# --- PROCESAMIENTO ---

bordes_sobel = detectar_bordes_sobel(imagen_original)
bordes_cerrados = cierre_morfologico_manual(bordes_sobel)
mascara_segmentada = rellenar_segmento_manual(bordes_cerrados)
imagen_segmentada = superponer_segmentacion(imagen_original, mascara_segmentada)

# --- VISUALIZACIÓN FINAL ---
cv2.imshow("Paso 5 - Imagen Original", imagen_original)
cv2.imshow("Paso 6 - Resultado Final: Segmentación superpuesta", imagen_segmentada)

cv2.waitKey(0)
cv2.destroyAllWindows()
