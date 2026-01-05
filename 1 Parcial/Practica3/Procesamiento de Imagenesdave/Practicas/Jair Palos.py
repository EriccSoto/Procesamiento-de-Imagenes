"""
UNIVERSIDAD AUTONOMA DEL ESTADO DE MEXICO

Centro Universitario UAEM Zumpango

Undad de Aprendizaje:Procesamiento de Imagenes

Autor: Jair Antonio Palos Hernadez

Descripcion: Codificar 18 Operaciones vistas en clases

Created: on tues Mar 08 5:56:44 2022

"""


from __future__ import division
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np


def Traslacion():
    ejex=50
    ejey=50
    image = cv2.imread('Baboon.png')
    ancho = image.shape[1] #columnas
    alto = image.shape[0] # filas
    # Traslación
    M = np.float32([[1,0,ejex],[0,1,ejey]])
    imageOut = cv2.warpAffine(image,M,(ancho,alto))
    print("Imagen trasladada, vaya a la imagen......")
    cv2.imshow('Imagen de entrada',image)
    cv2.imshow('Imagen de salida',imageOut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def  Escalado():
    image = cv2.imread('Barbara.png')
    # Escalando una imagen
    imageOut = cv2.resize(image,(600,300), interpolation=cv2.INTER_CUBIC)
    print("Imagen escalada, vaya a la imagen......")
    cv2.imshow('Imagen de entrada',image)
    cv2.imshow('Imagen de salida',imageOut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def  Rotacion():
    image  = cv2.imread('Baboon.png')
    rows,cols = image.shape[:2]
    # Rotación
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1.0)
    M1 = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
    M2 = cv2.getRotationMatrix2D((cols/2,rows/2),60,1)
    imageOut = cv2.warpAffine(image, M,(cols,rows) )
    imageOut1 = cv2.warpAffine(image, M1,(cols,rows) )
    imageOut2 = cv2.warpAffine(image, M2,(cols,rows) )
    print("Imagen rotada, vaya a la imagen......")
    cv2.imshow('Imagen de entrada',image)
    cv2.imshow('Imagen de salida 1',imageOut)
    cv2.imshow('Imagen de salida 2',imageOut1)
    cv2.imshow('Imagen de salida 3',imageOut2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def  RotacionPivote():
    image  = cv2.imread('ima1.jpg')
    rows,cols = image.shape[:2]
    # Rotación
    M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1.0)
    imageOut = cv2.warpAffine(image, M,(cols,rows) )
    print("Imagen rotada, vaya a la imagen......")
    cv2.imshow('Imagen de entrada',image)
    cv2.imshow('Imagen de salida 1',imageOut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Reflexion():
    image = cv2.imread('ima1.jpg')
    flip1 = cv2.flip(image,1)
    print("Imagen ya tine reflexion, vaya a la imagen......")
    cv2.imshow('Imagen de entrada',image)
    cv2.imshow('Imagen reflejada',flip1)
    cv2.waitKey(0)

def ReflexionX():
    imagen = cv2.imread('ima1.jpg')
    flip0 = cv2.flip(imagen,0)
    print("Imagen ya tine reflexion en X, vaya a la imagen......")
    cv2.imshow('Imagen de entrada',imagen)
    cv2.imshow('Imagen con reflexion en X',flip0)
    cv2.waitKey(0)

def ReflexionY():
    image = cv2.imread('ima1.jpg')
    flip1 = cv2.flip(image,1)
    print("Imagen ya tine reflexion Y, vaya a la imagen......")
    cv2.imshow('Imagen de entrada',image)
    cv2.imshow('Imagen reflejada',flip1)
    cv2.waitKey(0)

def ReflexionXY():
    imagen = cv2.imread('ima1.jpg')
    flip_1 = cv2.flip(imagen,-1)
    print("Imagen ya tiene reflexion en XY, vaya a la imagen......")
    cv2.imshow('Imagen de entrada',imagen)
    cv2.imshow('Imagen reflejada',flip_1)
    cv2.waitKey(0)
def Copia():
    img = cv2.imread('ima1.jpg',0)    
    filas, columnas = img.shape                                                                          
    copia = np.zeros((filas, columnas), dtype=np.uint8)                                                                                               
    for a in range(0, filas):  
        for b in range(0, columnas):                                                                                 
                 copia[a,b] =  img[a,b]
    print("Ya se hizo la copia, vaya a la imagen......")              
    cv2.imshow('Imagen de entrada', img)       
    cv2.imshow('Copia', copia)       
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Inversion():
    img = cv2.imread('ima1.jpg',0)    
    filas, columnas = img.shape                                                                          
    negativo = np.zeros((filas, columnas), dtype=np.uint8)                                                                                               
    for a in range(0, filas):  
        for b in range(0, columnas):                                                                                 
                 negativo[a,b] = 255 - img[a,b]
    print("Ya se aplico inversion, vaya a la imagen......")              
    cv2.imshow('Imagen de entrada', img)       
    cv2.imshow('Inversion', negativo)       
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Brillo():
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
    print("Ya se realizo la modificacion del brillo......")
    cv2.imshow('Imagen Original', imagen)
    cv2.imshow('Imagen Arreglada', ima)
    cv2.waitKey()

def Suma():
    img1=cv2.imread('Lena.png') 
    img2=cv2.imread('Baboon.png')
    img1_1=cv2.resize(img1,(300,300),interpolation= cv2.INTER_CUBIC) 
    img2_2=cv2.resize(img2,(300,300),interpolation= cv2.INTER_CUBIC)
    #Suma 
    suma=cv2.add(img1_1,img2_2)
    print("Ya se sumaron, vaya a la imagen......")
    cv2.imshow('Imagen 1',img1_1) 
    cv2.imshow('Imagen 2',img2_2) 
    cv2.imshow('Suma',suma)                                                 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Resta():
    img1=cv2.imread('ima1.jpg') 
    img2=cv2.imread('ima2.jpg')
    img1_1=cv2.resize(img1,(300,300),interpolation= cv2.INTER_CUBIC) 
    img2_2=cv2.resize(img2,(300,300),interpolation= cv2.INTER_CUBIC)
    #Resta
    resta=cv2.subtract(img1_1,img2_2)
    print("Ya se restaron, vaya a la imagen......")
    cv2.imshow('Imagen 1',img1_1) 
    cv2.imshow('Imagen 2',img2_2) 
    cv2.imshow('Resta',resta)                                                 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Multiplicacion():
    img1=cv2.imread('ima1.jpg') 
    img2=cv2.imread('ima2.jpg')
    img1_1=cv2.resize(img1,(300,300),interpolation= cv2.INTER_CUBIC) 
    img2_2=cv2.resize(img2,(300,300),interpolation= cv2.INTER_CUBIC)
    #Multiplicacion
    multi=cv2.multiply(img1_1,img2_2)
    print("Ya se multiplicaron, vaya a la imagen......")
    cv2.imshow('Imagen 1',img1_1) 
    cv2.imshow('Imagen 2',img2_2) 
    cv2.imshow('Multiplicacion',multi)                                                 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Division():
    img1=cv2.imread('ima1.jpg') 
    img2=cv2.imread('ima2.jpg')
    img1_1=cv2.resize(img1,(300,300),interpolation= cv2.INTER_CUBIC) 
    img2_2=cv2.resize(img2,(300,300),interpolation= cv2.INTER_CUBIC)
    #Division
    division=cv2.divide(img1_1,img2_2)
    print("Ya se dividieron, vaya a la imagen......")
    cv2.imshow('Imagen 1',img1_1) 
    cv2.imshow('Imagen 2',img2_2) 
    cv2.imshow('Division',division)                                                 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def AND1():
    img1=cv2.imread('ima1.jpg') 
    img2=cv2.imread('ima2.jpg')
    img1_1=cv2.resize(img1,(300,300),interpolation= cv2.INTER_CUBIC) 
    img2_2=cv2.resize(img2,(300,300),interpolation= cv2.INTER_CUBIC)
    #AND
    and1=cv2.bitwise_and(img1_1,img2_2)
    print("Ya se relaizo la operacion AND, vaya a la imagen......")
    cv2.imshow('Imagen 1',img1_1) 
    cv2.imshow('Imagen 2',img2_2) 
    cv2.imshow('AND',and1)                                                 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def OR1():
    img1=cv2.imread('ima1.jpg') 
    img2=cv2.imread('ima2.jpg')
    img1_1=cv2.resize(img1,(300,300),interpolation= cv2.INTER_CUBIC) 
    img2_2=cv2.resize(img2,(300,300),interpolation= cv2.INTER_CUBIC)
    #OR
    or1=cv2.bitwise_or(img1_1,img2_2)
    print("Ya se relaizo la operacion OR, vaya a la imagen......")
    cv2.imshow('Imagen 1',img1_1) 
    cv2.imshow('Imagen 2',img2_2) 
    cv2.imshow('OR',or1)                                                 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def NOT1():
    img1=cv2.imread('ima1.jpg') 
    img2=cv2.imread('ima2.jpg')
    img1_1=cv2.resize(img1,(300,300),interpolation= cv2.INTER_CUBIC) 
    img2_2=cv2.resize(img2,(300,300),interpolation= cv2.INTER_CUBIC)
    cv2.imshow('Imagen 1',img1_1) 
    cv2.imshow('Imagen 2',img2_2)  
    #NOT
    not1=cv2.bitwise_not(img1_1,img2_2)
    print("Ya se relaizo la operacion NOT, vaya a la imagen......")
    cv2.imshow('NOT',not1)                                                 
    cv2.waitKey(0)
    cv2.destroyAllWindows()


os.system('clear') # NOTA para windows tienes que cambiar clear por cls
print ("Selecciona una opción")
print ("\t1 - Traslacion")
print ("\t2 - Escalado")
print ("\t3 - Rotacion")
print ("\t4 - Rotacion Pivote")
print ("\t5 - Reflexion")
print ("\t6 - Reflexion en X")
print ("\t7 - Reflexion en Y")
print ("\t8 - Reflexion en XY")
print ("\t9- Copia")
print ("\t10- Inversion")
print ("\t11 - Brillo")
print ("\t12 - Suma")
print ("\t13-  Restas")
print ("\t14 - Multiplicacion")
print ("\t15 - Division")
print ("\t16-  AND")
print ("\t17 - OR")
print ("\t18 - NOT")
print ("\t19 - salir")

while True:
    opcionMenu = input("inserta la opción >> ")
    
    if opcionMenu=="1":
        print ("")
        Traslacion()
    elif opcionMenu=="2":
        print ("")
        Escalado()
    elif opcionMenu =="3":
        print ("")
        Rotacion()
    elif opcionMenu=="4":
        RotacionPivote()
        print ("")
    elif opcionMenu=="5":
        print ("")
        Reflexion()
    elif opcionMenu =="6":
        print ("")
        ReflexionX()
    elif opcionMenu=="7":
        print ("")
        ReflexionY()
    elif opcionMenu=="8":
        print ("")
        ReflexionXY()
    elif opcionMenu =="9":
        print ("")
        Copia()
    elif opcionMenu=="10":
        print ("")
        Inversion()
    elif opcionMenu=="11":
        print ("")
        Brillo()
    elif opcionMenu =="12":
        print ("")
        Suma()
    elif opcionMenu=="13":
        print ("")
        Resta()
    elif opcionMenu=="14":
        print ("")
        Multiplicacion()
    elif opcionMenu =="15":
        print ("")
        Division()
    elif opcionMenu=="16":
        print ("")
        AND1()
    elif opcionMenu=="17":
        print ("")
        OR1()
    elif opcionMenu =="18":
        print ("")
        NOT1()
    elif opcionMenu=="19":
        break
	
    else:
        print ("")
        input("No has pulsado ninguna opción correcta...\npulsa una tecla para continuar")


