import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------- FUNCIÓN: Cargar imagen ejemplo (camera) ---------
def cargar_imagen_camera():
    # Sustituyendo data.camera() con carga directa
    return cv2.imread(cv2.samples.findFile("dog3.jpg"), cv2.IMREAD_GRAYSCALE)

# --------- FUNCIÓN: Calcular histograma manual ----------
def calcular_histograma(imagen):
    h, w = imagen.shape
    hist = [0] * 256
    for y in range(h):
        for x in range(w):
            hist[int(imagen[y, x])] += 1
    return hist

# --------- FUNCIÓN: Calcular Multi-Otsu (dos umbrales = 3 clases) ---------
def calcular_multiotsu(hist, total_pixeles):
    max_varianza = 0
    umbral1, umbral2 = 0, 0

    for t1 in range(1, 255):
        for t2 in range(t1 + 1, 256):
            # Primer grupo: 0..t1
            w0 = sum(hist[0:t1])
            mu0 = sum(i * hist[i] for i in range(0, t1)) / w0 if w0 > 0 else 0

            # Segundo grupo: t1..t2
            w1 = sum(hist[t1:t2])
            mu1 = sum(i * hist[i] for i in range(t1, t2)) / w1 if w1 > 0 else 0

            # Tercer grupo: t2..255
            w2 = sum(hist[t2:256])
            mu2 = sum(i * hist[i] for i in range(t2, 256)) / w2 if w2 > 0 else 0

            # Pesos relativos
            P0 = w0 / total_pixeles
            P1 = w1 / total_pixeles
            P2 = w2 / total_pixeles

            mu_total = P0 * mu0 + P1 * mu1 + P2 * mu2

            varianza_entre = P0 * (mu0 - mu_total) ** 2 + \
                             P1 * (mu1 - mu_total) ** 2 + \
                             P2 * (mu2 - mu_total) ** 2

            if varianza_entre > max_varianza:
                max_varianza = varianza_entre
                umbral1 = t1
                umbral2 = t2

    return umbral1, umbral2

# --------- FUNCIÓN: Asignar regiones manualmente ---------
def asignar_regiones(imagen, t1, t2):
    h, w = imagen.shape
    salida = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            val = imagen[y, x]
            if val < t1:
                salida[y, x] = 0  # clase 0
            elif val < t2:
                salida[y, x] = 1  # clase 1
            else:
                salida[y, x] = 2  # clase 2
    return salida

# --------- MAIN ---------
imagen = cargar_imagen_camera()
hist = calcular_histograma(imagen)
t1, t2 = calcular_multiotsu(hist, imagen.shape[0] * imagen.shape[1])
regiones = asignar_regiones(imagen, t1, t2)

# --------- VISUALIZACIÓN ---------
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

# Imagen original
ax[0].imshow(imagen, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

# Histograma con umbrales
ax[1].bar(range(256), hist, color='gray')
ax[1].set_title('Histograma con umbrales')
ax[1].axvline(t1, color='r', linestyle='--', label=f'T1 = {t1}')
ax[1].axvline(t2, color='g', linestyle='--', label=f'T2 = {t2}')
ax[1].legend()

# Imagen segmentada
ax[2].imshow(regiones, cmap='jet')
ax[2].set_title('Resultado Multi-Otsu')
ax[2].axis('off')

plt.tight_layout()
plt.show()
