import cv2
import numpy as np

def leer_imagen_binaria(ruta):
    """Lee una imagen en escala de grises y la convierte en binaria."""
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    _, binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    return binaria

def seleccionar_kernel(img_binaria, x, y, k_alto, k_ancho):
    """Selecciona una submatriz (patrón) de la imagen binaria para usar como kernel."""
    return img_binaria[y:y+k_alto, x:x+k_ancho]

def erosion_manual_con_kernel(imagen, kernel):
    """Realiza la erosión manual utilizando el kernel."""
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
    """Aplica erosión utilizando el kernel de 'Hit'."""
    return erosion_manual_con_kernel(imagen, kernel)

def erosion_manual_miss(imagen, kernel):
    """Aplica erosión utilizando el complemento del kernel de 'Miss'."""
    complemento = cv2.bitwise_not(imagen)
    return erosion_manual_con_kernel(complemento, kernel)

def hit_or_miss_manual(imagen, kernel_hit, kernel_miss):
    """Realiza la operación de 'Hit or Miss' manual con dos kernels."""
    hit = erosion_manual_hit(imagen, kernel_hit)
    miss = erosion_manual_miss(imagen, kernel_miss)
    resultado = cv2.bitwise_and(hit, miss)
    return resultado

if __name__ == "__main__":
    # Leer la imagen binaria
    ruta_imagen = 'patron.png'  # Cambia la ruta si es necesario
    img_binaria = leer_imagen_binaria(ruta_imagen)

    # Selección del kernel a partir de la imagen binaria (por ejemplo, seleccionando una submatriz 5x5)
    x, y = 10, 10  # Coordenadas de la esquina superior izquierda de la submatriz
    k_alto, k_ancho = 5, 5  # Tamaño del kernel (puedes ajustar el tamaño)

    kernel = seleccionar_kernel(img_binaria, x, y, k_alto, k_ancho)

    # Mostrar el kernel seleccionado
    print("Kernel seleccionado:")
    print(kernel)

    # Realizar la operación 'Hit or Miss' utilizando el kernel
    hitmiss_manual = hit_or_miss_manual(img_binaria, kernel, kernel)

    # Mostrar solo la imagen binaria original y el resultado de "Hit or Miss Manual"
    cv2.imshow('Binaria', img_binaria)
    cv2.imshow('Hit or Miss Manual', hitmiss_manual)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
