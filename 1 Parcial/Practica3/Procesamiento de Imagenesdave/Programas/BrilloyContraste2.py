##Chavez Zamorano Cesar
##Navarro Campos Marissa Belen
##Palos Hernandez Jair Antonio 

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

imagen = cv.imread('Paisaje.jpg')## se Ingresa la imagen
if imagen is None:## en este if nos da mencion si la imagen maraca none nos manda un mensaje de que no se pudo abrir la imagen
    print('No se puede abrir la imagen: ')
    exit(0)
ima = np.zeros(imagen.shape, imagen.dtype)##se hace una copia del la imahen con su mismo tamaño y forma
contraste = float (input ( 'Ingrese el valor del Contraste en el intervalo 1.0 al 3.0:  ' ))## se manda a pedir un numero flotante para el contraste
## se manda a pedir un numero entero para el brillo que no pase del intervalo establecido, ya que se veria tortalmente en blanco o en negro
brillo = int (input ( 'Ingrese el valor del Brillo en el intervalo -100 al 100:  ' )) 

for y in range(imagen.shape[0]):## Se hace el for para el tamaño de las columnas
    for x in range(imagen.shape[1]):## Se hace el for para el tamaño de las filas
        for c in range(imagen.shape[2]):
            ima[y,x,c] = np.clip(contraste*imagen[y,x,c] + brillo, 0, 255)## se hace la representacionde la ecuacion para mejorar el brillo y el contraste
cv.imshow('IMagen Original', imagen)## se muestra la imagen original
cv.imshow('Imagen Arreglada', ima)## me muestra la imagen arreglada
cv.waitKey()


def  hisgrama (img):## e hace una funcion
	alto, ancho = img.shape[0:2]
	#crear matrices para el histograma
	hb1= np.zeros((256,1))## se guada en ceros con esas dimensiones
	hg= np.zeros((256,1))## se guada en ceros con esas dimensiones
	hr= np.zeros((256,1))## se guada en ceros con esas dimensiones
	##clacular los pixeles de cada intensidad
	for i in range(alto):##recorrido en lo alto
    		for j in range (ancho):##recorrido en lo ancho
    			b = img.item(i, j, 2)## se guardan las intensidades para el azul
    			g = img.item(i, j, 1)## se guardan las intensidades para el verde
    			r = img.item(i, j, 0)## se guardan las intensidades para el rojo
    			hb1[b] = hb1[b]+1 ## se guada la intensidad en el arreglo
    			hg[g] = hg[g]+1 ## se guada la intensidad en el arreglo
    			hr[r] = hr[r]+1 ## se guada la intensidad en el arreglo        	    

	plt.plot(hb1,color = 'b')## para graficar el histograma de color azul
	plt.title("Azul")#titulo azul
	plt.xlim([0,256])##los rango
	plt.show()## mostrar el histograma

	plt.plot(hg,color = 'g')## para graficar el histograma de color verde
	plt.title("Verde")#titulo verde
	plt.xlim([0,256])##los rango
	plt.show()## mostrar el histograma

	plt.plot(hr,color = 'r')##para graficar el hstograma de color rojo
	plt.title("Rojo")##Titulo en verde
	plt.xlim([0,256])##los rangos
	plt.show()## mostrar el histograma


ht=hisgrama( imagen )## se llama la funcion para el imagen original
ht2=hisgrama( ima )## se llama la funcion para el imagen areglada
