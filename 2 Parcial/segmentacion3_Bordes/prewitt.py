import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- FUNCIONES DE CONVOLUCIÓN Y BORDES ---
def convolucion(imagen, kernel):
    h, w = imagen.shape
    kh, kw = kernel.shape
    kh2, kw2 = kh // 2, kw // 2
    salida = np.zeros_like(imagen, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            suma = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    i = y + ky - kh2
                    j = x + kx - kw2
                    if 0 <= i < h and 0 <= j < w:
                        suma += imagen[i, j] * kernel[ky, kx]
            salida[y, x] = suma
    return salida

# --- CARGA Y PREWITT ---
imagen_color = cv2.imread("dog4.jpg")
imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

kernel_prewitt_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])

kernel_prewitt_y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]])

edges_x = convolucion(imagen_gris.astype(np.float32), kernel_prewitt_x)
edges_y = convolucion(imagen_gris.astype(np.float32), kernel_prewitt_y)
edges = np.sqrt(edges_x**2 + edges_y**2).astype(np.uint8)

# --- BINARIZACIÓN Y MORFOLOGÍA (CIERRE) ---
_, bordes_bin = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
kernel_morf = np.ones((5, 5), np.uint8)
cerrado = cv2.morphologyEx(bordes_bin, cv2.MORPH_CLOSE, kernel_morf)

# --- CONTORNO MÁS EXTERNO ---
contornos, _ = cv2.findContours(cerrado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mascara = np.zeros_like(imagen_gris)
cv2.drawContours(mascara, contornos, -1, 255, -1)  # Relleno blanco sobre fondo negro

# --- SUPERPONER SEGMENTACIÓN EN COLOR VERDE TRANSPARENTE ---
segmentado = imagen_color.copy()
for y in range(mascara.shape[0]):
    for x in range(mascara.shape[1]):
        if mascara[y, x] == 255:
            b, g, r = segmentado[y, x]
            segmentado[y, x] = [
                int(b * 0.5 + 0.5 * 0),
                int(g * 0.5 + 0.5 * 255),
                int(r * 0.5 + 0.5 * 0)
            ]

# --- MOSTRAR RESULTADOS ---
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.imshow(imagen_gris, cmap='gray')
plt.title("Imagen original gris")

plt.subplot(1, 3, 2)
plt.imshow(cerrado, cmap='gray')
plt.title("Bordes cerrados (Prewitt + cierre)")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(segmentado, cv2.COLOR_BGR2RGB))
plt.title("Segmentación resaltada")
plt.tight_layout()
plt.show()
