from ipaddress import summarize_address_range
import math
from ssl import _create_default_https_context
from matplotlib import pyplot as plt
import os
from PIL import Image
import cv2
import numpy as np

def HistogramaGris2(ima):
    
    x = np.linspace(0, 255, num=256, dtype=np.uint8)
    y =np.zeros(256)

    for w in range (ima.shape[0]):
     for h in range (ima.shape[1]):
        v = ima[w, h]
        y[v] = y[v]+1
    plt.bar(x, y)
    plt.title("Histograma de Brillo")
    plt.show()
   

def HistogramaGris():
    img= cv2.imread("ima1.jpg",cv2.IMREAD_GRAYSCALE)
    width,height = img.shape
    cv2.imshow("Imagen", img)
    x = np.linspace(0, 255, num=256, dtype=np.uint8)
    y =np.zeros(256)

    for w in range (width):
     for h in range (height):
        v = img[w, h]
        y[v] = y[v] + 1
    plt.bar(x, y)
    plt.title("Escala de gris")
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def HistogramaNomrmalizado():
    img = cv2.imread("ima1.jpg",cv2.IMREAD_GRAYSCALE)
    width,height= img.shape
    cv2.imshow("Imagen",img)
    x = np.linspace(0,255,num=256,dtype=np.uint8)
    y = np.zeros(256)
    tamano = 1/(width*height)
    
    for w in range (width):
        for h in range (height):
            v=img[w,h]
            y[v]=(y[v]+1*tamano)     

    plt.bar(x,y)
    plt.title("Normalizado")
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def HistogramaExpansion():
    gray_img = Image.open('ima1.jpg').convert("LA")
    row = gray_img.size[0]
    col = gray_img.size[1]
    img2 = Image.new("L", (row, col))
    high = 0
    low = 255
    MAX = 500
    MIN = 0
    for x in range(0 , row):
        for y in range(0, col):
            valor = gray_img.getpixel((x,y))[0]
            if high <  valor:
                high = valor
            if low > valor:
                low = valor
    
    for x in range(0 , row):
        for y in range(0, col):
            valor = int((((gray_img.getpixel((x,y))[0]-low)/(high-low))*(MAX-MIN)) + MIN)
            img2.putpixel((x,y), valor)



    y = img2.histogram()
    y = y[:-1]
    x = np.round(np.linspace(MIN, MAX, len(y)))
    plt.title("Expansion del Histograma")
    plt.bar(x, y)
    plt.show()

def HistogramaContraccion():
    gray_img = Image.open('ima1.jpg').convert("LA")
    row = gray_img.size[0]
    col = gray_img.size[1]
    img2 = Image.new("L", (row, col))
    high = 255
    low = 0
    MAX=200
    MIN=100
    for x in range(1 , row):
        for y in range(1, col):
            if high < gray_img.getpixel((x,y))[0] :
                high = gray_img.getpixel((x,y))[0]
            if low > gray_img.getpixel((x,y))[0]:
                low = gray_img.getpixel((x,y))[0]
    for x in range(1 , row):
        for y in range(1, col):
            img2.putpixel((x,y), int((((MAX-MIN)/(high-low))*(gray_img.getpixel((x,y))[0]-low))+MIN))
            
    y = img2.histogram()
    x = np.arange(len(y))
    plt.title("Contraccion del Histograma")
    plt.bar(x, y)
    plt.show()

def HistogramaDesplazamiento():
    gray_img = Image.open('ima1.jpg').convert("LA")
    row = gray_img.size[0]
    col = gray_img.size[1]
    img2 = Image.new("L", (row, col))
    high = 0
    low = 255
    desplazamiento = 20
    for x in range(0 , row):
        for y in range(0, col):
            valor = gray_img.getpixel((x,y))[0]
            if high <  valor:
                high = valor
            if low > valor:
                low = valor

    for x in range(0 , row):
        for y in range(0, col):
            valor = int((gray_img.getpixel((x,y))[0])+ desplazamiento)
            img2.putpixel((x,y), valor)
    y = img2.histogram()
    y = y[:-1]
    x = np.arange(len(y))
    plt.title("Desplazamineto del Histograma")
    plt.bar(x, y)
    plt.show()

def HistogramaBrillo():
    imagen = cv2.imread('ima1.jpg')
    if imagen is None:
        print('No se puede abrir la imagen: ')
        exit(0)
    ima = np.zeros(imagen.shape, imagen.dtype)
    contraste = float (input ( '* Ingrese el valor del Contraste en el intervalo 1.0 al 3.0: ' ))
    brillo = int (input ( '* Ingrese el valor del Brillo en el intervalo -100 al 100: ' )) 
    for y in range(imagen.shape[0]):
        for x in range(imagen.shape[1]):
            for c in range(imagen.shape[2]):
                ima[y,x,c] = np.clip(contraste*imagen[y,x,c] + brillo, 0, 255)
    HistogramaGris2(ima)
    print("Ya se realizo la modificacion del brillo......")
    cv2.imshow('Imagen Original', imagen)
    cv2.imshow('Imagen Arreglada', ima)
    cv2.waitKey()
    

def  Ecualizacion():
    
    img = cv2.imread('ima1.jpg')
    cv2.imshow('ima1.jpg', img)
    color = ('b','g','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    plt.show()
    cv2.destroyAllWindows()

def extra():
    imagen = input("Ingresa el nombre de la imagen: ")
    img = plt.imread(imagen)

    print( img.shape )
    print( img.size )
    alto, ancho = img.shape[0:2]
    print(alto)
    print(ancho)


        #canales con los que estoy trabajando
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]

    #Mostrar las imagenes con intensidades RGB
    plt.imshow(red)
    plt.title("Rojo")
    plt.axis("off")
    plt.show()
    #
    plt.imshow(green)
    plt.title("Verde")
    plt.axis("off")
    plt.show()
    #
    plt.imshow(blue)
    plt.title("Azul")
    plt.axis("off")
    plt.show()


    #crear matrices para el histograma
    hb= np.zeros((256,1))
    hg= np.zeros((256,1))
    hr= np.zeros((256,1))

    #clacular los pixeles de cada intensidad
    for i in range(alto):
        for j in range (ancho):
            b = img.item(i, j, 2)
            g = img.item(i, j, 1)
            r = img.item(i, j, 0)
            hb[b] = hb[b]+1
            hg[g] = hg[g]+1
            hr[r] = hr[r]+1

    plt.plot(hb,color = 'b')
    plt.title("Azul")
    plt.xlim([0,256])
    plt.show()

    plt.plot(hg,color = 'g')
    plt.title("Verde")
    plt.xlim([0,256])
    plt.show()

    plt.plot(hr,color = 'r')
    plt.title("Rojo")
    plt.xlim([0,256])
    plt.show()


    plt.hist( blue.ravel(), bins=256 )  #8by    
    plt.title("azul")
    plt.show()

    plt.hist( green.ravel(), bins=256 )  #8by
    plt.title("Verde")
    plt.show()

    plt.hist( red.ravel(), bins=256 )  #8by
    plt.title("Rojo 256")
    plt.show()

os.system('clear') # NOTA para windows tienes que cambiar clear por cls

print ("Selecciona una opción")
print ("\t1 - Histograma a Gris")
print ("\t2 - Histograma Normalizado")
print ("\t3 - Histograma  Expansion")
print ("\t4 - Histograma  Contraccion")
print ("\t5 - Histograma  Desplazamiento")
print ("\t6 - Histograma  Brillo")
print ("\t7 - Ecyalizacion del Histograma")
print ("\t8 - Extra--- histograma de colores  Brillo")

while True:
    opcionMenu = input("inserta la opción >> ")
    
    if opcionMenu=="1":
        HistogramaGris()
    if opcionMenu=="2":
        HistogramaNomrmalizado()
    if opcionMenu=="3":
        HistogramaExpansion()
    if opcionMenu=="4":
        HistogramaContraccion()
    if opcionMenu=="5":
        HistogramaDesplazamiento()
    if opcionMenu=="6":
        HistogramaBrillo()
    elif opcionMenu=="7":
        Ecualizacion()  
    elif opcionMenu=="8":
        extra()     
    else:
        input("No has pulsado ninguna opción correcta...\npulsa una tecla para continuar")