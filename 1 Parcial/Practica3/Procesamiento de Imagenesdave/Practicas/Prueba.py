import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import random

def GaussieNoisy(image,sigma):
    #    
    img = image.astype(np.int16)  # Este paso es evitar el caso donde el punto de píxel sea inferior a 0, más de 255
    mu = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img[i, j, k] = img[i, j, k] + random.gauss(mu=mu, sigma=sigma)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img

def spNoisy(image,s_vs_p = 0.5,amount = 0.004):
    #  
    out = np.copy(image)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 1
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out




def ArithmeticMeanAlogrithm(image):
    
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] = np.mean(image[i-1:i+2,j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)
def rgbArithmeticMean(image):
    r,g,b = cv2.split(image)
    r = ArithmeticMeanAlogrithm(r)
    g = ArithmeticMeanAlogrithm(g)
    b = ArithmeticMeanAlogrithm(b)
    return cv2.merge([r,g,b])


def GeometricMeanOperator(roi):
    roi = roi.astype(np.float64)
    p = np.prod(roi)
    return p ** (1 / (roi.shape[0] * roi.shape[1]))
def GeometricMeanAlogrithm(image):
    # Filtrado medio geométrico
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = GeometricMeanOperator(image[i - 1:i + 2, j - 1:j + 2])
    new_image = (new_image - np.min(image)) * (255 / np.max(image))
    return new_image.astype(np.uint8)
def rgbGemotriccMean(image):
    r,g,b = cv2.split(image)
    r = GeometricMeanAlogrithm(r)
    g = GeometricMeanAlogrithm(g)
    b = GeometricMeanAlogrithm(b)
    return cv2.merge([r,g,b])


def HarmonicMeanOperator(roi):
    roi = roi.astype(np.float64)
    if 0 in roi:
        roi = 0
    else:
        roi = scipy.stats.hmean(roi.reshape(-1))
    return roi
def HarmonicMeanAlogrithm(image):
    # Filtro medio armónico
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] =HarmonicMeanOperator(image[i-1:i+2,j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)
def rgbHarmonicMean(image):
    r,g,b = cv2.split(image)
    r = HarmonicMeanAlogrithm(r)
    g = HarmonicMeanAlogrithm(g)
    b = HarmonicMeanAlogrithm(b)
    return cv2.merge([r,g,b])


def Contra_harmonicMeanOperator(roi,q):
    roi = roi.astype(np.float64)
    return np.mean((roi)**(q+1))/np.mean((roi)**(q))
def Contra_harmonicMeanAlogrithm(image,q):
    
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] = Contra_harmonicMeanOperator(image[i-1:i+2,j-1:j+2],q)
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)
def rgbContra_harmonicMean(image,q):
    r,g,b = cv2.split(image)
    r = Contra_harmonicMeanAlogrithm(r,q)
    g = Contra_harmonicMeanAlogrithm(g,q)
    b = Contra_harmonicMeanAlogrithm(b,q)
    return cv2.merge([r,g,b])




if __name__ == '__main__':
    house = cv2.imread("lena.png")
    house = cv2.resize(cv2.cvtColor(house, cv2.COLOR_BGR2RGB), (200, 200))
    plt.imshow(house)
    plt.axis("off")
    plt.title("Original Image")
    plt.show()   #      
    
    flagN = input("Por favor, seleccione el ruido unido: \t "
                  "Ruido gaussiano - 1 \t"
                  "Ruido de pimienta - 2 \t")

    if flagN == "1":
        GuassHouse = GaussieNoisy(house,18)
        plt.imshow(GuassHouse)
        plt.axis("off")
        plt.title("Gauss noise Image")
        plt.show()   # Añadir imagen después del ruido gaussiano
    elif flagN == "2":
        spHouse = spNoisy(house)
        plt.imshow(spHouse)
        plt.axis("off")
        plt.title("Salt And peper Image")
        plt.show()  #            


    flagF = input("Por favor, seleccione el filtro: \ N"
                  "Filtro medio aritmético - A \ N"
                  "Filtro medio geométrico - b \ n"
                  "Filtro medio armónico - C \ N"
                  "Filtro medio armónico inverso - D \ N")

    if flagF == "a":
        if flagN == "1":
            plt.imshow(rgbArithmeticMean(GuassHouse))
        elif flagN == "2":
            plt.imshow(rgbArithmeticMean(spHouse))
        plt.title("Arithmetic Mean Filter")
        plt.show()  # Arithmetic Mean Filter
    elif flagF == "b":
        if flagN == "1":
            plt.imshow(rgbGemotriccMean(GuassHouse))
        elif flagN == "2":
            plt.imshow(rgbGemotriccMean(spHouse))
        plt.title("Geometric Mean Filter")
        plt.show()  # Geometric Mean Filter
    elif flagF == "c":
        if flagN == "1":
            plt.imshow(rgbHarmonicMean(GuassHouse))
        elif flagN == "2":
            plt.imshow(rgbHarmonicMean(spHouse))
        plt.title("Harmonic Mean Filter")
        plt.show()  # Harmonic Mean Filter
    elif flagF == "d":
        if flagN == "1":
            plt.imshow(rgbContra_harmonicMean(GuassHouse,2))
        elif flagN == "2":
            plt.imshow(rgbContra_harmonicMean(spHouse,2))
        plt.title("Contra-harmonic Mean Filter")
        plt.show()  # Contra-harmonic Mean Filter

