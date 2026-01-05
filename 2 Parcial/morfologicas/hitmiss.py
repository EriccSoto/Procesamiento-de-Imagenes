import cv2
import numpy as np

def leer_imagen_binaria(ruta):
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    _, binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    return binaria

def erosion_manual_con_kernel(imagen, kernel, titulo="Erosión"):
    alto, ancho = imagen.shape
    k_alto, k_ancho = kernel.shape
    pad_y, pad_x = k_alto // 2, k_ancho // 2

    relleno = np.zeros((alto + 2 * pad_y, ancho + 2 * pad_x), dtype=np.uint8)
    relleno[pad_y:pad_y+alto, pad_x:pad_x+ancho] = imagen

    resultado = np.zeros_like(imagen)

    for y in range(alto):
        for x in range(ancho):
            ventana = relleno[y:y+k_alto, x:x+k_ancho]
            coincide = True
            for i in range(k_alto):
                for j in range(k_ancho):
                    if kernel[i, j] == 1 and ventana[i, j] != 255:
                        coincide = False
            resultado[y, x] = 255 if coincide else 0

    return resultado

def erosion_manual_hit(imagen, kernel):
    return erosion_manual_con_kernel(imagen, kernel, "Erosión - HIT")

def erosion_manual_miss(imagen, kernel):
    complemento = cv2.bitwise_not(imagen)
    return erosion_manual_con_kernel(complemento, kernel, "Erosión - MISS")

def hit_or_miss_manual(imagen, kernel_hit, kernel_miss):
    hit = erosion_manual_hit(imagen, kernel_hit)
    miss = erosion_manual_miss(imagen, kernel_miss)
    resultado = cv2.bitwise_and(hit, miss)
    return resultado

if __name__ == "__main__":
    ruta_imagen = 'patron.png'  # Cambia la ruta si es necesario
    img_binaria = leer_imagen_binaria(ruta_imagen)

    kernel_hit = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]], dtype=np.uint8)

    kernel_miss = np.array([[1, 1, 0, 1, 1],
                        [1, 0, 0, 0, 1],
                        [0, 0, 1, 0, 0],
                        [1, 0, 0, 0, 1],
                        [1, 1, 0, 1, 1]], dtype=np.uint8)

    # Aplicar la operación manual de "Hit or Miss"
    hitmiss_manual = hit_or_miss_manual(img_binaria, kernel_hit, kernel_miss)

    # Mostrar solo la imagen binaria y el resultado de "Hit or Miss Manual"
    cv2.imshow('Binaria', img_binaria)
    cv2.imshow('Hit or Miss Manual', hitmiss_manual)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
