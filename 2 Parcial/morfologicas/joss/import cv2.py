import cv2
import numpy as np

def erode_manual(image, kernel):
    img_height, img_width = image.shape
    k_height, k_width = kernel.shape
    k_h_margin = k_height // 2
    k_w_margin = k_width // 2
    output_image = np.zeros_like(image)
    
    for y in range(k_h_margin, img_height - k_h_margin):
        for x in range(k_w_margin, img_width - k_w_margin):
            submatrix = image[y - k_h_margin:y + k_h_margin + 1, x - k_w_margin:x + k_w_margin + 1]
            if np.all(submatrix == 255):
                output_image[y, x] = 255
                
    return output_image

def dilate_manual(image, kernel):
    img_height, img_width = image.shape
    k_height, k_width = kernel.shape
    k_h_margin = k_height // 2
    k_w_margin = k_width // 2
    output_image = np.zeros_like(image)
    
    for y in range(k_h_margin, img_height - k_h_margin):
        for x in range(k_w_margin, img_width - k_w_margin):
            submatrix = image[y - k_h_margin:y + k_h_margin + 1, x - k_w_margin:x + k_w_margin + 1]
            if np.any(submatrix == 255):
                output_image[y, x] = 255
                
    return output_image

def gradient_manual(dilated, eroded):
    return cv2.absdiff(dilated, eroded)

if __name__ == '__main__':
    img_binaria = cv2.imread('flor.png', cv2.IMREAD_GRAYSCALE)
    
    if img_binaria is None:
        print("Error al cargar la imagen.")
        exit()
    
    _, img_binaria = cv2.threshold(img_binaria, 127, 255, cv2.THRESH_BINARY)
    
    cv2.imshow('Imagen Binaria Original', img_binaria)
    
    kernel = np.ones((7, 7), np.uint8)
    
    img_erosionada = erode_manual(img_binaria, kernel)
    img_dilatada = dilate_manual(img_binaria, kernel)
    img_gradiente = gradient_manual(img_dilatada, img_erosionada)
    
    cv2.imshow('Imagen Erosionada', img_erosionada)
    cv2.imshow('Imagen Dilatada', img_dilatada)
    cv2.imshow('Imagen Gradiente', img_gradiente)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
