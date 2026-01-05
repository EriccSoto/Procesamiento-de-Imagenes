import cv2
import numpy as np

def leer_imagen_binaria(ruta):
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    _, binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    return binaria

def skeleton_manual(imagen_binaria):
    
    imagen = imagen_binaria.copy()
    skel = np.zeros(imagen.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        erosion = cv2.erode(imagen, kernel)
        apertura = cv2.dilate(erosion, kernel)
        temp = cv2.subtract(imagen, apertura)
        skel = cv2.bitwise_or(skel, temp)
        imagen = erosion.copy()
        if cv2.countNonZero(imagen) == 0:
            break

    return skel

def skeleton_opencv(imagen_binaria):
    
    try:
        from cv2.ximgproc import thinning
        return thinning(imagen_binaria)
    except:
        print("Instala con: pip install opencv-contrib-python")
        return np.zeros_like(imagen_binaria)

if __name__ == "__main__":
    ruta_imagen = 'estrella.png'
    imagen_binaria = leer_imagen_binaria(ruta_imagen)

    # Esqueleto manual paso a paso
    esqueleto_manual = skeleton_manual(imagen_binaria)

    # Esqueleto con OpenCV si está disponible
    esqueleto_cv = skeleton_opencv(imagen_binaria)

    # Mostrar resultados
    cv2.imshow('1 - Imagen binaria original', imagen_binaria)
    cv2.imshow('2 - Esqueleto manual', esqueleto_manual)
    cv2.imshow('3 - Esqueleto OpenCV', esqueleto_cv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
