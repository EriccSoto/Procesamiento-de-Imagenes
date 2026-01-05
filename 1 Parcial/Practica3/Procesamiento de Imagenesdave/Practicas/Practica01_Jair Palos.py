"""
UNIVERSIDAD AUTONOMA DEL ESTADO DE MEXICO

Centro Universitario UAEM Zumpango

Undad de Aprendizaje:Procesamiento de Imagenes

Autor: Jair Antonio Palos Hernadez

Descripcion: Codificar 18 Operaciones vistas en clases

Created: on tues Mar 08 5:56:44 2022

"""
from __future__ import division
import math
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np

def Traslacion():
    img = cv2.imread('Lena.png',0)    
    imgInfo = img.shape
    filas= imgInfo[0]
    columnas = imgInfo[1]
    x_T=50
    y_T=50                                                                          
    copia = np.zeros((filas, columnas), dtype=np.uint8)                                                                                               
    for a in range(filas):  
        for b in range(columnas):
             if(a+x_T)<filas and(b+y_T)<columnas:                                                                              
                copia[a + x_T,b+y_T] =  img[a,b]
    print("Ya se hizo la traslacion, vaya a la imagen......")              
    cv2.imshow('Imagen de entrada', img)       
    cv2.imshow('Copia', copia)       
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def  Escalado():
    x_T=2
    y_T=2
    img = cv2.imread('Barbara.png')    
    filas, columnas, ncolores = img.shape                                                                          
    copia = np.zeros((int(filas*x_T),int(columnas*y_T),ncolores))                                                                                               
    for a in range(filas):  
        for b in range(columnas):                                                                                 
                 copia[int(a*x_T),int(b*y_T),: ] =  img[a,b,:]
    print("Ya se hizo el escalado, vaya a la imagen......")              
    cv2.imshow('Imagen de entrada', img)       
    cv2.imshow('Copia', copia)       
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def  Rotacion():
    img = cv2.imread('Baboon.png')    
    alto, ancho, ncolores = img.shape 
    angulo = math.pi/4
    copia = np.zeros((int(alto),int(ancho),ncolores))
    for y in range(alto):  
        for x in range(ancho):
            xr=abs (int(x*math.cos(angulo)-y*math.sin(angulo)))
            yr=abs (int(y*math.cos(angulo)-x*math.sin(angulo)))
            if xr >0 and yr>0 and xr < ancho and yr<alto:
                copia[yr,xr,: ] =  img[y,x,:]
    print("Ya se hizo la rotacion, vaya a la imagen......")              
    cv2.imshow('Imagen de entrada', img)        
    cv2.imshow('Copia', copia)       
    cv2.waitKey(0)
    cv2.destroyAllWindows()
       
def  RotacionPivote():
    img = cv2.imread('ima2.jpg')    
    alto, ancho, ncolores = img.shape 
    angulo = math.pi/4
    copia = np.zeros((int(alto),int(ancho),ncolores))
    for y in range(alto):  
        for x in range(ancho):
            xr=abs (int(x*math.cos(angulo)-y*math.sin(angulo)))
            yr=abs (int(y*math.cos(angulo)-x*math.sin(angulo)))
            if xr >0 and yr>0 and xr < ancho and yr<alto:
                copia[yr,xr,: ] =  img[y,x,:]
    print("Ya se hizo la rotacion, vaya a la imagen......")              
    cv2.imshow('Imagen de entrada', img)        
    cv2.imshow('Copia', copia)       
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def Reflexion():
    img = cv2.imread('ima1.jpg',1)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    img2 = np.zeros(img.shape,np.uint8)
    for i in range(height):
        for j in range(width):
            img2[i,j] = img[i,width -1- j]
    print("Imagen con reflexion, vaya a la imagen......")
    cv2.imshow('img',img2)
    cv2.imshow('Imagen de eontrada',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ReflexionX():
    img = cv2.imread('ima1.jpg',1)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    img2 = np.zeros(img.shape,np.uint8)
    for i in range(height):
        for j in range(width):
            img2[i,j] = img[height-1 - i,j]
    print("Imagen con reflexion, vaya a la imagen......")
    cv2.imshow('img',img2)
    cv2.imshow('Imagen de entrada',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def ReflexionY():
    img = cv2.imread('ima1.jpg',1)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    img2 = np.zeros(img.shape,np.uint8)
    for i in range(height):
        for j in range(width):
            img2[i,j] = img[i,width -1- j]
    print("Imagen con reflexion, vaya a la imagen......")
    cv2.imshow('img',img2)
    cv2.imshow('Imagen de entrada',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ReflexionXY():
    img = cv2.imread('ima1.jpg',1)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    img2 = np.zeros(img.shape,np.uint8)
    dst = np.zeros((height*2,width,3),np.uint8)
    for i in range(height):
        for j in range(width):
            img2[i,j] = img[height-1 - i,width-1-j]
    print("Imagen rotada, vaya a la imagen......")
    cv2.imshow('img',img2)
    cv2.imshow('Imagen de entrada',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    img1=cv2.imread('ima1.jpg') 
    img2=cv2.imread('ima2.jpg')
    imgInfo = img1.shape
    height = imgInfo[0]
    width = imgInfo[1]
    s = np.zeros(img1.shape,np.uint8)
    for f in range( height ):
        for c in range( width ):
            s[f,c]  = img2[f,c] + img1[f,c]
    print("Ya se sumaron, vaya a la imagen......")
    cv2.imshow('Imagen 1',img1) 
    cv2.imshow('Imagen 2',img2) 
    cv2.imshow('Suma',s)                                                 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Resta():
    img1=cv2.imread('ima1.jpg') 
    img2=cv2.imread('ima2.jpg')
    imgInfo = img1.shape
    height = imgInfo[0]
    width = imgInfo[1]
    s = np.zeros(img1.shape,np.uint8)
    for f in range( height ):
        for c in range( width ):
            s[f,c]  = img1[f,c] - img2[f,c]
    print("Ya se restaron, vaya a la imagen......")
    cv2.imshow('Imagen 1',img1) 
    cv2.imshow('Imagen 2',img2) 
    cv2.imshow('Resta',s)                                                 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Multiplicacion():
    img1=cv2.imread('ima1.jpg') 
    img2=cv2.imread('ima2.jpg')
    imgInfo = img1.shape
    height = imgInfo[0]
    width = imgInfo[1]
    s = np.zeros(img1.shape,np.uint8)
    for f in range( height ):
        for c in range( width ):
            s[f,c]  = img1[f,c] * img2[f,c]
    print("Ya se multiplicaron, vaya a la imagen......")
    cv2.imshow('Imagen 1',img1) 
    cv2.imshow('Imagen 2',img2) 
    cv2.imshow('Multiplicacion',s)                                                 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Division():
    img1=cv2.imread('ima1.jpg') 
    img2=cv2.imread('ima2.jpg')
    imgInfo = img1.shape
    height = imgInfo[0]
    width = imgInfo[1]
    s = np.zeros(img1.shape,np.uint8)
    for f in range( height ):
        for c in range( width ):
            s[f,c]  = img1[f,c] / img2[f,c]
    print("Ya se multiplicaron, vaya a la imagen......")
    cv2.imshow('Imagen 1',img1) 
    cv2.imshow('Imagen 2',img2) 
    cv2.imshow('Division',s)                                                 
    cv2.waitKey(0)
    cv2.destroyAllWindows()  

def AND1():
    img1=cv2.imread('ima1.jpg') 
    img2=cv2.imread('ima2.jpg')
    imgInfo = img1.shape
    height = imgInfo[0]
    width = imgInfo[1]
    s = np.zeros(img1.shape,np.uint8)
    for f in range( height ):
        for c in range( width ):
            s[f,c]  = img2[f,c] & img1[f,c]
    print("Ya se aplico en AND, vaya a la imagen......")
    cv2.imshow('Imagen 1',img1) 
    cv2.imshow('Imagen 2',img2) 
    cv2.imshow('AND',s)                                                 
    cv2.waitKey(0)
    cv2.destroyAllWindows()  


def OR1():
    img1=cv2.imread('ima1.jpg') 
    img2=cv2.imread('ima2.jpg')
    imgInfo = img1.shape
    height = imgInfo[0]
    width = imgInfo[1]
    s = np.zeros(img1.shape,np.uint8)
    for f in range( height ):
        for c in range( width ):
            s[f,c]  = img1[f,c] | img2[f,c]
    print("Ya se  aplico el OR, vaya a la imagen......")
    cv2.imshow('Imagen 1',img1) 
    cv2.imshow('Imagen 2',img2) 
    cv2.imshow('OR',s)                                                 
    cv2.waitKey(0)
    cv2.destroyAllWindows()  


def NOT1():
    img1=cv2.imread('ima1.jpg') 
    imgInfo = img1.shape
    height = imgInfo[0]
    width = imgInfo[1]
    s = np.zeros(img1.shape,np.uint8)
    for f in range( height ):
        for c in range( width ):
            s[f,c]  != img1[f,c]
    print("Ya se  aplico el OR, vaya a la imagen......")
    cv2.imshow('Imagen 1',img1) 
    cv2.imshow('OR',s)                                                 
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