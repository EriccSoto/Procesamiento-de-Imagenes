import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- CARGA Y PREPROCESADO ----------
def cargar_imagen_color(ruta):
    return cv2.imread(ruta, cv2.IMREAD_COLOR)

def convertir_a_grises_manual(imagen):
    h, w, _ = imagen.shape
    gris = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            b = imagen[y, x][0]
            g = imagen[y, x][1]
            r = imagen[y, x][2]
            gris[y, x] = 0.114 * b + 0.587 * g + 0.299 * r
    return gris

def suavizar_manual(imagen_gris):
    kernel = [
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ]
    h, w = imagen_gris.shape
    salida = np.zeros((h, w), dtype=np.float32)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            suma = 0.0
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    pixel = imagen_gris[y + ky, x + kx]
                    peso = kernel[ky + 1][kx + 1]
                    suma += pixel * peso
            salida[y, x] = min(max(suma, 0), 255)
    
    salida_entera = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            salida_entera[y, x] = int(salida[y, x])
    return salida_entera

# ---------- HISTOGRAMA ----------
def calcular_histograma(imagen):
    h, w = imagen.shape
    hist = [0] * 256
    for y in range(h):
        for x in range(w):
            val = int(imagen[y, x])
            hist[val] += 1
    return hist

def mostrar_histograma(hist, titulo):
    x = list(range(256))
    y = hist
    plt.figure(figsize=(10, 4))
    plt.title(titulo)
    plt.xlabel("Valor de gris")
    plt.ylabel("Frecuencia")
    plt.plot(x, y, color='black')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------- SUPERPOSICIÓN ----------
def superponer_transparente(imagen, mascara, alpha=0.5, color=(0, 255, 0)):
    salida = imagen.copy()
    h, w = mascara.shape
    for y in range(h):
        for x in range(w):
            if mascara[y, x] == 255:
                b = salida[y, x][0]
                g = salida[y, x][1]
                r = salida[y, x][2]
                b_new = int((1 - alpha) * b + alpha * color[0])
                g_new = int((1 - alpha) * g + alpha * color[1])
                r_new = int((1 - alpha) * r + alpha * color[2])
                salida[y, x][0] = min(b_new, 255)
                salida[y, x][1] = min(g_new, 255)
                salida[y, x][2] = min(r_new, 255)
    return salida

# ---------- SEGMENTACIÓN POR UMBRAL FIJO ----------
def segmentar_umbral_absoluto(imagen_gris, umbral):
    h, w = imagen_gris.shape
    salida = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            salida[y, x] = 255 if imagen_gris[y, x] >= umbral else 0
    return salida

# ---------- MAIN ----------
ruta = "dog4.jpg"
img_color = cargar_imagen_color(ruta)
img_gris = convertir_a_grises_manual(img_color)
img_suavizada = suavizar_manual(img_gris)

# Histograma antes
hist_pre = calcular_histograma(img_suavizada)
mostrar_histograma(hist_pre, "Histograma antes de binarización (imagen suavizada)")

# Segmentación por umbral fijo
umbral_fijo = 90  # Puedes ajustar este valor
mask_umbral = segmentar_umbral_absoluto(img_suavizada, umbral_fijo)

# Histograma después
hist_post = calcular_histograma(mask_umbral)
mostrar_histograma(hist_post, "Histograma después de binarización (imagen binaria con umbral fijo)")

# Mostrar resultados
cv2.imshow("Original", img_color)
cv2.imshow("Umbral Fijo - Mascara", mask_umbral)
cv2.imshow("Umbral Fijo - Superpuesto", superponer_transparente(img_color, mask_umbral, alpha=0.5))
cv2.waitKey(0)
cv2.destroyAllWindows()
