import os
import math
from statistics import median
from PIL import Image, ImageFilter
import PIL.Image
import cv2
import numpy as np
import random
import scipy
import scipy.stats
from matplotlib import pyplot as plt
from tkinter import *
from skimage import io
from skimage.util import random_noise
from skimage import feature, filters
import copy
import imutils
import pylab
##___________________Gausiana_______________________________
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
##____________________Sal___________________________________
def sal():
    img=cv2.imread("lena.png")
    percentage = 0.05
    num=int(percentage*img.shape[0]*img.shape[1])# Número de puntos de ruido de sal y pimienta
    random.randint(0, img.shape[0])
    img2=img.copy()
    for i in range(num):
        X=random.randint(0,img2.shape[0]-1)# Un número entero aleatorio desde 0 hasta la longitud de la imagen, porque es un intervalo cerrado, -1
        Y=random.randint(0,img2.shape[1]-1)
        if random.randint(0,1) ==0: # Probabilidad en blanco y negro 55 abierto
            img2[X,Y] = (255,255,255)#blanco        
    cv2.imshow("lena",img)
    cv2.imshow("Sal",img2)
    cv2.waitKey(0)
##____________________Pimienta_______________________________
def pimienta():
    img=cv2.imread("lena.png")
    percentage = 0.05
    num=int(percentage*img.shape[0]*img.shape[1])# Número de puntos de ruido de sal y pimienta
    random.randint(0, img.shape[0])
    img2=img.copy()
    for i in range(num):
        X=random.randint(0,img2.shape[0]-1)# Un número entero aleatorio desde 0 hasta la longitud de la imagen, porque es un intervalo cerrado, -1
        Y=random.randint(0,img2.shape[1]-1)
        if random.randint(0,1) ==0: # Probabilidad en blanco y negro 55 abierto
            img2[X,Y] = (0,0,0)#blanco        
    cv2.imshow("lena",img)
    cv2.imshow("Pimienta",img2)
    cv2.waitKey(0)
##____________________Sal y Pimienta_________________________
def salYpimienta():

    img=cv2.imread("lena.png")
    percentage = 0.05
    num=int(percentage*img.shape[0]*img.shape[1])# Número de puntos de ruido de sal y pimienta
    random.randint(0, img.shape[0])
    img2=img.copy()
    for i in range(num):
        X=random.randint(0,img2.shape[0]-1)# Un número entero aleatorio desde 0 hasta la longitud de la imagen, porque es un intervalo cerrado, -1
        Y=random.randint(0,img2.shape[1]-1)
        if random.randint(0,1) ==0: # Probabilidad en blanco y negro 55 abierto
            img2[X,Y] = (255,255,255)#blanco
        else:
            img2[X,Y] =(0,0,0)#negro
    cv2.imshow("lena",img)
    cv2.imshow("Sal",img2)
    cv2.waitKey(0)
def sYp(img):

    
    percentage = 0.04
    num=int(percentage*img.shape[0]*img.shape[1])# Número de puntos de ruido de sal y pimienta
    random.randint(0, img.shape[0])
    img2=img.copy()
    for i in range(num):
        X=random.randint(0,img2.shape[0]-1)# Un número entero aleatorio desde 0 hasta la longitud de la imagen, porque es un intervalo cerrado, -1
        Y=random.randint(0,img2.shape[1]-1)
        if random.randint(0,1) ==0: # Probabilidad en blanco y negro 55 abierto
            img2[X,Y] = (255,255,255)#blanco
        else:
            img2[X,Y] =(0,0,0)#negro
    return img2

##______________________________________________________________________________________________________

##____________________Aritmetica_____________________________
def ArithmeticMeanAlogrithm(image):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] = np.mean(image[i-1:i+2,j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    print(new_image.astype(np.uint8))
    return new_image.astype(np.uint8)
def rgbArithmeticMean(img):
    
    r,g,b = cv2.split(img)
    r = ArithmeticMeanAlogrithm(r)
    g = ArithmeticMeanAlogrithm(g)
    b = ArithmeticMeanAlogrithm(b)
    return cv2.merge([r,g,b])

##____________________Geometrica_____________________________
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
    print(new_image.astype(np.uint8))
    return new_image.astype(np.uint8)
def rgbGemotriccMean(image):
    r,g,b = cv2.split(image)
    r = GeometricMeanAlogrithm(r)
    g = GeometricMeanAlogrithm(g)
    b = GeometricMeanAlogrithm(b)
    
    return cv2.merge([r,g,b])
    
##____________________Armonica_______________________________
def HarmonicMeanOperator(roi):
    roi = roi.astype(np.float64)
    if 0 in roi:
        roi = 0
    else:
        roi = scipy.stats.hmean(roi.reshape(-1))
      ##  print(roi)
    return roi
def HarmonicMeanAlogrithm(image):
    # Filtro medio armónico
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] =HarmonicMeanOperator(image[i-1:i+2,j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    print(new_image.astype(np.uint8))
    return new_image.astype(np.uint8)
def rgbHarmonicMean(image):
   
    r,g,b = cv2.split(image)
    r = HarmonicMeanAlogrithm(r)
    g = HarmonicMeanAlogrithm(g)
    b = HarmonicMeanAlogrithm(b)
    print(cv2.merge([r,g,b]))
    return cv2.merge([r,g,b])
    
##____________________Armonica inversa_______________________________
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
def rgbContra_harmonicMean(image):
    q=2
    r,g,b = cv2.split(image)
    r = Contra_harmonicMeanAlogrithm(r,q)
    g = Contra_harmonicMeanAlogrithm(g,q)
    b = Contra_harmonicMeanAlogrithm(b,q)
    print(cv2.merge([r,g,b]))
    return cv2.merge([r,g,b])
   
##______________________________________________________________________________________________________

##___________________Mediana__________________________________________
def mediana(img):
    m, n = img.shape 
    img1 = np.zeros([m, n])   
    for i in range(1, m-1): 
        for j in range(1, n-1): 
            temp = [img[i-1, j-1], 
                img[i-1, j], 
                img[i-1, j + 1], 
                img[i, j-1], 
                img[i, j], 
                img[i, j + 1], 
                img[i + 1, j-1], 
                img[i + 1, j], 
                img[i + 1, j + 1]] 
            temp = sorted(temp) 
            img1[i, j]= temp[4] 
    img1 = img1.astype(np.uint8) 
    return img1
def Rmediana(image):
    r,g,b = cv2.split(image)
    r = mediana(r)
    g = mediana(g)
    b = mediana(b)
    print(cv2.merge([r,g,b]))
    return cv2.merge([r,g,b])
##___________________Maxima___________________________________________
def spilt( a ):
    if a/2 == 0:
        x1 = x2 = a/2
    else:
        x1 = math.floor( a/2 )
        x2 = a - x1
    return -x1,x2
def original (i, j, k,a, b,img):
    x1, x2 = spilt(a)
    y1, y2 = spilt(b)
    temp = np.zeros(a * b)
    count = 0
    for m in range(x1, x2):
        for n in range(y1, y2):
            if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:
                temp[count] = img[i, j, k]
            else:
                temp[count] = img[i + m, j + n, k]
            count += 1
    return  temp 
def max_functin(a, b, img):
    img0 = copy.copy(img)
    for i in range(0, img.shape[0]):
        for j in range(2, img.shape[1]):
            for k in range(img.shape[2]):
                temp = original(i, j, k, a, b, img0)
                img[i, j, k] = np.max(temp)
    print(img)
    return img
##__________________Minima____________________________________________
# Filtrado mínimo
def min_functin(a, b, img):
    img0 = copy.copy(img)
    for i in range(0, img.shape[0]):
        for j in range(2, img.shape[1]):
            for k in range(img.shape[2]):
                temp = original(i, j, k, a, b, img0)
                img[i, j, k] = np.min(temp)
    print(img)
    return img
##_________________Punto Medio____________________________________________________
# Filtrado medio
def mid_functin(a, b, img):
    img0 = copy.copy(img)
    for i in range(0, img.shape[0]):
        for j in range(2, img.shape[1]):
            for k in range(img.shape[2]):
                temp = original(i, j, k, a, b, img0)
                img[i, j, k] = np.median(temp)
    print(img)
    return img

##__________________Local adaptativo______________________________________________
def local():
    img  = cv2.imread('lena.png', 0)
    img_1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 11, 2)
    plt.imshow(img_1,'gray')
    plt.title("Local Adaptativo")
    plt.show() 
##__________________medio adaptivo_________________________________________________
def medio():
    img  = cv2.imread('lena.png', 0)
    img_1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 11, 2)
    plt.imshow(img_1,'gray')
    plt.title("Medio Adaptativo")
    plt.show() 
##___________________Gaussian_______________________________________________________
def gaussian():
    img  = cv2.imread('lena.png', 0)
    img_1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 11, 2)
    plt.imshow(img_1,'gray')
    plt.title("Gaussian")
    plt.show()
##___________________Transformacion Gamma______________________________________________

def gamma_trans(img,gamma):
	 # El método específico se normaliza primero a 1, y luego se usa gamma como valor de índice para encontrar el nuevo valor de píxel y luego restaurar
	gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
	 # La implementación de mapeo utiliza la función de búsqueda de tablas de Opencv
	return cv2.LUT(img0,gamma_table)

##_________________Destacar Niveles de gris
def gris():
    rojoBajo1 = np.array([0, 140, 90], np.uint8)
    rojoAlto1 = np.array([8, 255, 255], np.uint8)
    rojoBajo2 = np.array([160, 140, 90], np.uint8)
    rojoAlto2 = np.array([180, 255, 255], np.uint8)
    # Leer la imagen
    image = cv2.imread('ima4.jpg')
    image = imutils.resize(image, width=640)
    # Pasamos las imágenes de BGR a: GRAY (esta a BGR nuevamente) y a HSV
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageGray = cv2.cvtColor(imageGray, cv2.COLOR_GRAY2BGR)
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Detectamos el color rojo
    maskRojo1 = cv2.inRange(imageHSV, rojoBajo1, rojoAlto1)
    maskRojo2 = cv2.inRange(imageHSV, rojoBajo2, rojoAlto2)
    mask = cv2.add(maskRojo1,maskRojo2)
    mask = cv2.medianBlur(mask, 7)
    redDetected = cv2.bitwise_and(image,image,mask=mask)
    # Fondo en grises
    invMask = cv2.bitwise_not(mask)
    bgGray = cv2.bitwise_and(imageGray,imageGray,mask=invMask)
    # Sumamos bgGray y redDetected
    finalImage = cv2.add(bgGray,redDetected)
    # Visualización
    cv2.imshow('Image',image)
    cv2.imshow('Gris total', bgGray)
    cv2.imshow('Final', finalImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
##________________planos de bits__________________________________________________
def planos():
    lena=cv2.imread("lena.png",0)
    cv2.imshow("lena",lena)
    r, c = lena.shape # El rango de la imagen leída
    x=np.zeros((r,c,8),dtype=np.uint8)
    for i in range(8):
         x [:,:, i] = 2 ** i # significa 2 elevado a i
    r=np.zeros((r,c,8),dtype=np.uint8)
    for i in range(8):
        r[:,:,i]=cv2.bitwise_and(lena, x[:,:,i])
        mask=r[:,:,i]>0
        r [mask] = 255 # Establezca el valor de píxel mayor que 0 a 255, el propósito es aumentar el brillo, si el valor es demasiado bajo, estará cerca del negro
    cv2.imshow(str(i),r[:,:,i])
    cv2.waitKey()
    cv2.destroyAllWindows()

##_________________convoluciones________________________________________________________

##_________________ Convolucion Este____________________________________________________
def este():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
 
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
 
    fil = np.array ([[1, 0, -1], # Este es el filtro establecido, que es el núcleo de convolución
                [ 2, 0, -2],
                [  1, 0, -1]])
 
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
 
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()

##_________________Convolucion Norte____________________________________________________
def norte():
    img = plt.imread ("A.jpg") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([[-1, -2, -1], # Este es el filtro establecido, que es el núcleo de convolución
                     [ 2, 0, -2],
                     [  1, 2, 1]])
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()
##_________________Convolucion Noreste__________________________________________________
def noreste():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([[0, -1, -2], # Este es el filtro establecido, que es el núcleo de convolución
                     [ 1, 0, -1],
                     [  2, 1, 0]])
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()
##________________Convolucion Noroeste__________________________________________________
def noroeste():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([[-2, -1, 0], # Este es el filtro establecido, que es el núcleo de convolución
                     [ 1, 0, -1],
                     [  2, 1, 0]])
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()
##_________________Convolucion Oeste______________________________________________________
def oeste():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([[-1, 0, 1], # Este es el filtro establecido, que es el núcleo de convolución
                     [ -2, 0, 2],
                     [  -1, 0, 1]])
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()
##_________________Convolucion Sur______________________________________________________
def sur():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([[1, 2, 1], # Este es el filtro establecido, que es el núcleo de convolución
                     [ 0, 0, 0],
                     [  -1, -2, -1]])
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()
def sureste():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([[1, 2, 1], # Este es el filtro establecido, que es el núcleo de convolución
                     [ 0, 0, 0],
                     [  -1, -2, -1]])
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()

##_________________Convolucion Suroeste______________________________________________________
def suroeste():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([[0, 1, 2], # Este es el filtro establecido, que es el núcleo de convolución
                     [-1,0, 1],
                     [-2,-1,0]])
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()

def pasaaltas3x3():
    ruta = ("Lena.png")
    im = cv2.imread( ruta,cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Normal', im)
    a = [ [ -1.0, -1.0, -1.0 ],
           [ -1.0, 9.0, -1.0 ],
           [ -1.0, -1.0, -1.0 ] ]
    kernel = np.asarray(a)
    dst = cv2.filter2D(im, -1, kernel)
    cv2.imshow('Pasa altas 1', dst)
    cv2.waitKey(0)
def pasabajas3x3():
    img = plt.imread ("A.jpg") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([ [ 1.0, 2.0, 1.0 ],
           [ 2.0, 4.0, 2.0 ],
           [ 1.0, 2.0, 1.0 ] ])
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()
def lap3x3():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([ [ 0.0, -1.0, 0.0 ],
           [ -1.0, 4.0, -1.0 ],
           [ 0.0, -1.0, 0.0 ] ])
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()

def robert():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([ [ 0.0, -1.0],
                      [ -1.0, 0.0] ], dtype=int)
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()
def robertX():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([ [ 0.0, -1.0],
                      [ 1.0, 0.0] ], dtype=int)
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()
def robertY():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([ [ -1.0, 0.0],
                      [ 0.0, 1.0] ], dtype=int)
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()
def lap5x5():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([ [-1.0,-1.0,-1.0,-1.0,-1.0],
                      [-1.0,-1.0,-1.0,-1.0,-1.0],
                      [-1.0,-1.0, 24 ,-1.0,-1.0],
                      [-1.0,-1.0,-1.0,-1.0,-1.0],
                      [-1.0,-1.0,-1.0,-1.0,-1.0] ], dtype=int)
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()
def pasaaltas5x5():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([ [0.0,-1.0,-1.0,-1.0,0.0],
                      [-1.0,2.0,-4.0,2.0,-1.0],
                      [-1.0,-4.0,13.0,-4.0,-1.0],
                      [-1.0,2.0,-4.0,2.0,-1.0],
                      [0.0,-1.0,-1.0,-1.0,0.0] ], dtype=int)
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()
def pasabajas5x5():
    img = plt.imread ("A.jpg") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([ [1.0,1.0,1.0,1.0,1.0],
                      [1.0,4.0,4.0,4.0,1.0],
                      [1.0,4.0,12.0,4.0,1.0],
                      [1.0,4.0,4.0,4.0,1.0],
                      [1.0,1.0,1.0,1.0,1.0] ], dtype=int)
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()
def Gauss7x7():
    img = plt.imread ("Lena.png") #Lea la imagen aquí
    plt.imshow (img) # Muestra la imagen leída
    pylab.show()
    fil = np.array ([ [1.0,1.0,2.0,2.0,2.0,1.0,1.0],
                      [1.0,2.0,2.0,4.0,2.0,2.0,1.0],
                      [2.0,2.0,4.0,8.0,4.0,2.0,2.0],
                      [2.0,4.0,8.0,16.0,8.0,4.0,2.0],
                      [2.0,2.0,4.0,8.0,4.0,2.0,2.0],
                      [1.0,2.0,2.0,4.0,2.0,2.0,1.0],
                      [1.0,1.0,2.0,2.0,2.0,1.0,1.0] ], dtype=int)
    res = cv2.filter2D (img, -1, fil) #Utilice la función de convolución de opencv
    plt.imshow (res) #Muestra la imagen después de la convolución
    pylab.show()
#_____________________________________________________________________________________________________________________
def candy():
    img = cv2.imread('Lena.png', 0)
    edges_canny = cv2.Canny(img, 100, 100)

    cv2.imshow('Image',img)
    cv2.imshow('Candy', edges_canny)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

##__________________Sobel_______________________________________________________________
def sobel():
    img = cv2.imread('A.jpg', 0)
    edges_sobel = filters.sobel(img) # Sobel
    cv2.imshow('Image',img)
    cv2.imshow('Sobel',edges_sobel )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def lap():
    img = cv2.imread('Lena.png', 0)
    edges_sobel = filters.laplace(img) # Laplacian # Sobel
    cv2.imshow('Image',img)
    cv2.imshow('Lapciano',edges_sobel )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Scharr():
    img = cv2.imread('Lena.png', 0)
    edges_sobel = filters.scharr(img) # Scharr
    cv2.imshow('Image',img)
    cv2.imshow('Scharr',edges_sobel )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Prewitt():
    img = cv2.imread('Lena.png', 0)
    edges_sobel = filters.prewitt(img) # Prewitt
    cv2.imshow('Image',img)
    cv2.imshow('Prewitt',edges_sobel )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Roberts():
    img = cv2.imread('Lena.png', 0)
    edges_sobel =filters.roberts(img) # Roberts
    cv2.imshow('Image',img)
    cv2.imshow('Roberts',edges_sobel )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def SobelX():
    img = cv2.imread('Lena.png', 0)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    cv2.imshow('Image',img)
    cv2.imshow('Sobel X',sobel_x )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def SobelY():
    img = cv2.imread('Lena.png', 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    cv2.imshow('Image',img)
    cv2.imshow('Sobel Y',sobel_y )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def ScharrX():
    img = cv2.imread('Lena.png', 0)
    schar_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    cv2.imshow('Image',img)
    cv2.imshow('Scharr X',schar_x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def ScharrY():
    img = cv2.imread('Lena.png', 0)
    schar_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    cv2.imshow('Image',img)
    cv2.imshow('Scharr Y',schar_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def candyB():
    img = cv2.imread('Lena.png', 0)
    edges_canny = cv2.Canny(img, 100, 100)
    cv2.imshow('Image',img)
    cv2.imshow('Canny',edges_canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def lapB():
    img = cv2.imread('Lena.png', 0)
    edges_laplacian = cv2.Laplacian(img, cv2.CV_64F)
    cv2.imshow('Image',img)
    cv2.imshow('Lapciano',edges_laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#_________________________________________________________________________________
os.system('clear') # NOTA para windows tienes que cambiar clear por cls

print ("Selecciona una opción")
print ("\t---------Ruidos Solicitados------------")
print ("\t1 - Sal")
print ("\t2 - Pemienta")
print ("\t3 - Sal y Pemienta")
print ("\t---------Flitro de Media---------------")
print ("\t4 - Aritmetica")
print ("\t5 - Geometrico")
print ("\t6 - Harmonico")
print ("\t7 - Hamonico Inverso")
print("-------------------------------------------")
print ("\t8 - Mediana")
print ("\t9 - Maxima")
print ("\t10 - Minima")
print ("\t11 - Punto Medio")
print("-------------------------------------------")
print ("\t12 - Local adaptivo")
print ("\t13 - Media adaptivo")
print ("\t14 - Gaussiano")
print("-------------Modificadores de gris----------")
print ("\t15 - Transformacion Gamma")
print ("\t16 - Estiramiento de contraste")
print ("\t17 - Destacar niveles de gris")
print ("\t18 - Planos de bits")
print("-------------Convuluciones----------")
print ("\t19 - Convolucion Gradiante Este")
print ("\t20 - Convolucion Gradiante Norte")
print ("\t21 - Convolucion Gradiante Noreste")
print ("\t22 - Convolucion Gradiante Noroeste")
print ("\t23 - Convolucion Gradiante Sureste")
print ("\t24 - Convolucion Gradiante Suroeste")
print ("\t25 - Convolucion Gradiante Sur")
print ("\t26 - Convolucion Gradiante Oeste")
print ("\t27 - Gaussiana")
print ("\t28 - Pasaaltas 3x3")
print ("\t29 - Pasabajas 3x3")
print ("\t30 - Lapciano 3x3")
print ("\t31 - Sobel Vertical")
print ("\t32 - Sobel Horizontal")
print ("\t33 - robert")
print ("\t34 - Lapciano 5x5")
print ("\t35 - Pasaaltas 5x5")
print ("\t36 - Pasabajas 5x5")
print("-------------Bordes----------")
print ("\t37 - Candy")
print ("\t38 - Sobel")
print ("\t39 - Lapciano")
print ("\t40 - Scharr")
print ("\t41 - Prewitt")
print ("\t42 - Roberts")
print("-------------Otro tipos de Bordes----------")
print ("\t43 - Sobel x")
print ("\t44 - Sobel y")
print ("\t45 - Scharr x")
print ("\t46 - Scharr y")
print ("\t47 - Canny")
print ("\t48 - Lapciano")
print ("\t49 - Salir")
while True:
    opcionMenu = input("inserta la opción >> ")
    
    if opcionMenu=="1":
        sal()
    if opcionMenu=="2":
        pimienta()
    if opcionMenu=="3":
        salYpimienta()
    if opcionMenu=="4":

        ima = cv2.imread("lena.png")
        ima = cv2.resize(cv2.cvtColor(ima, cv2.COLOR_BGR2RGB), (200, 200))

        print("Ruido gaussiano - 1 ")
        print("Ruido de sal y pimienta - 2 ")
        flagN = input("Por favor, seleccione el ruido unido: ")

        if flagN == "1":
            G = GaussieNoisy(ima,18)
            plt.imshow(G)
            plt.axis("off")
            plt.title("Imagen Gaussiona")
            plt.show()   # Añadir imagen después del ruido gaussiano
            
        elif flagN == "2":
            G = sYp(ima)
            plt.imshow(G)
            plt.axis("off")
            plt.title("Imagen sal y pimienta")
            plt.show()  #
        plt.imshow(rgbArithmeticMean(G))
        plt.title("Aritmetico")
        plt.show()
    if opcionMenu=="5":
        ima = cv2.imread("lena.png")
        ima = cv2.resize(cv2.cvtColor(ima, cv2.COLOR_BGR2RGB), (200, 200))

        print("Ruido gaussiano - 1 ")
        print("Ruido de sal y pimienta - 2 ")
        flagN = input("Por favor, seleccione el ruido unido: ")

        if flagN == "1":
            G = GaussieNoisy(ima,18)
            plt.imshow(G)
            plt.axis("off")
            plt.title("Imagen Gaussiona")
            plt.show()   # Añadir imagen después del ruido gaussiano
            
        elif flagN == "2":
            G = sYp(ima)
            plt.imshow(G)
            plt.axis("off")
            plt.title("Imagen sal y pimienta")
            plt.show()  #
        plt.imshow(rgbGemotriccMean(G))
        plt.title("Geometrico")
        plt.show()        
    if opcionMenu=="6":
        ima = cv2.imread("lena.png")
        ima = cv2.resize(cv2.cvtColor(ima, cv2.COLOR_BGR2RGB), (200, 200))

        print("Ruido gaussiano - 1 ")
        print("Ruido de sal y pimienta - 2 ")
        flagN = input("Por favor, seleccione el ruido unido: ")

        if flagN == "1":
            G = GaussieNoisy(ima,18)
            plt.imshow(G)
            plt.axis("off")
            plt.title("Imagen Gaussiona")
            plt.show()   # Añadir imagen después del ruido gaussiano
            
        elif flagN == "2":
            G = sYp(ima)
            plt.imshow(G)
            plt.axis("off")
            plt.title("Imagen sal y pimienta")
            plt.show()  #
        plt.imshow(rgbHarmonicMean(G))
        plt.title("Harmonico")
        plt.show()        
       
    elif opcionMenu=="7":
        ima = cv2.imread("lena.png")
        ima = cv2.resize(cv2.cvtColor(ima, cv2.COLOR_BGR2RGB), (200, 200))

        print("Ruido gaussiano - 1 ")
        print("Ruido de sal y pimienta - 2 ")
        flagN = input("Por favor, seleccione el ruido unido: ")

        if flagN == "1":
            G = GaussieNoisy(ima,18)
            plt.imshow(G)
            plt.axis("off")
            plt.title("Imagen Gaussiona")
            plt.show()   # Añadir imagen después del ruido gaussiano
            
        elif flagN == "2":
            G = sYp(ima)
            plt.imshow(G)
            plt.axis("off")
            plt.title("Imagen sal y pimienta")
            plt.show()  #
        plt.imshow(rgbContra_harmonicMean(G))
        plt.title("Harmonico Inverso")
        plt.show()    
        
    if opcionMenu=="8":
        ima = cv2.imread("lena.png")
        ima = cv2.resize(cv2.cvtColor(ima, cv2.COLOR_BGR2RGB), (200, 200))
        print("Ruido gaussiano - 1 ")
        print("Ruido de sal y pimienta - 2 ")
        flagN = input("Por favor, seleccione el ruido unido: ")

        if flagN == "1":
            G = GaussieNoisy(ima,18)
            plt.imshow(G)
            plt.axis("off")
            plt.title("Imagen Gaussiona")
            plt.show()   # Añadir imagen después del ruido gaussiano
            
        elif flagN == "2":
            G = sYp(ima)
            plt.imshow(G)
            plt.axis("off")
            plt.title("Imagen sal y pimienta")
            plt.show()  #
        plt.imshow(Rmediana(G))
        plt.title("Mediana")
        plt.show()    
        
    if opcionMenu=="9":
        img0 = cv2.imread(r"Ruido.png")
        max_img = max_functin(3, 3, copy.copy(img0))
        cv2.imshow("original", img0)
        cv2.imshow("Maxima",max_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if opcionMenu=="10":
        img0 = cv2.imread(r"Ruido.png")
        min_img = min_functin(3, 3, copy.copy(img0))
        cv2.imshow("original", img0)
        cv2.imshow("Minima",min_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if opcionMenu=="11":
        img0 = cv2.imread(r"Ruido.png")
        mid_img = mid_functin(3, 3, copy.copy(img0))
        cv2.imshow("original", img0)
        cv2.imshow("Punto Medio",mid_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    if opcionMenu=="12":
        local()
    if opcionMenu=="13":
        medio()
    if opcionMenu=="14":
        gaussian()
    if opcionMenu=="15":
        
        img0 = cv2.imread('lena.png')
        img0_corrted = gamma_trans(img0, 0.5)
        cv2.imshow('Imagen Original',img0)
        cv2.imshow('Transfomracion Gamma',img0_corrted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if opcionMenu=="16":
        lena = cv2.imread("lena.png", 0)
        hist_full = cv2.calcHist([lena], [0], None, [256], [0, 256])
        Imax = np.max(lena)
        Imin = np.min(lena)
        MAX = 255
        MIN = 0
        mejorada = (lena - Imin) / (Imax - Imin) * (MAX - MIN) + MIN
        cv2.imshow("contraste estirado", mejorada.astype("uint8"))
        cv2.imshow("lena",lena)
        cv2.waitKey()
        mejo = cv2.calcHist([mejorada.astype("uint8")], [0], None, [256], [0, 256])
        plt.plot(hist_full)
        plt.title("Histograma de imagen normal")
        plt.show()
        plt.plot(mejo)
        plt.title("Histograma de estiramiento de contraste")
        plt.show()     
    if opcionMenu=="17":
        gris()
    if opcionMenu=="18":
        planos()
    if opcionMenu=="19":
        este()
    if opcionMenu=="20":
        norte()
    if opcionMenu=="21":
        noreste()
    if opcionMenu=="22":
        noroeste()
    if opcionMenu=="23":
        sur()
    if opcionMenu=="24":
        oeste()
    if opcionMenu=="25":
        sureste()
    if opcionMenu=="26":
        suroeste()
    if opcionMenu=="27":
        Gauss7x7()
    if opcionMenu=="28":
        pasaaltas3x3()
    if opcionMenu=="29":
        pasabajas3x3()
    if opcionMenu=="30":
        lap3x3()
    if opcionMenu=="31":
        norte()
    if opcionMenu=="32":
        este()
    if opcionMenu=="33":
        robert()
        robertX()
        robertY()
    if opcionMenu=="34":
        lap5x5()
    if opcionMenu=="35":
        pasaaltas5x5()
    if opcionMenu=="36":
        pasabajas5x5()
    if opcionMenu=="37":
        candy()
    if opcionMenu=="38":
        sobel()
    if opcionMenu=="39":
        lap()
    if opcionMenu=="40":
        Scharr()
    if opcionMenu=="41":
        Prewitt()
    if opcionMenu=="42":
        Roberts()
    if opcionMenu=="43":
        SobelX()
    if opcionMenu=="44":
        SobelY()
    if opcionMenu=="45":
        ScharrX()
    if opcionMenu=="46":
        ScharrY()
    if opcionMenu=="47":
        candyB()
    if opcionMenu=="48":
        lapB()
    if opcionMenu=="49":
        break
    else:
        input("No has pulsado ninguna opción correcta...\npulsa una tecla para continuar")