##Chavez Zamorano Cesar
##Navarro Campos Marissa Belen
##Palos Hernandez Jair Antonio 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

imagen = cv.imread('Manzana.jpg')## se Ingresa la imagen
if imagen is None:## en este if nos da mencion si la imagen maraca none nos manda un mensaje de que no se pudo abrir la imagen
    print('No se puede abrir la imagen: ')
    exit(0)
ima = np.zeros(imagen.shape, imagen.dtype)##se hace una copia del la imahen con su mismo tamaño y forma
contraste = float (input ( '* Ingrese el valor del Contraste en el intervalo 1.0 al 3.0:  ' ))## se manda a pedir un numero flotante para el contraste
## se manda a pedir un numero entero para el brillo que no pase del intervalo establecido, ya que se veria tortalmente en blanco o en negro
brillo = int (input ( '* Ingrese el valor del Brillo en el intervalo -100 al 100:  ' )) 

for y in range(imagen.shape[0]):## Se hace el for para el tamaño de las columnas
    for x in range(imagen.shape[1]):## Se hace el for para el tamaño de las filas
        for c in range(imagen.shape[2]):
            ima[y,x,c] = np.clip(contraste*imagen[y,x,c] + brillo, 0, 255)## se hace la representacionde la ecuacion para mejorar el brillo y el contraste
cv.imshow('IMagen Original', imagen)## se muestra la imagen original
cv.imshow('Imagen Arreglada', ima)## me muestra la imagen arreglada
cv.waitKey()


def histGris( im ):## cre Frea una funcion
    h = np.zeros([256]);## toda la matriz lllegan de ceros
    for f in im:##recorre lo alto
        for c in f:##recorre lo ancho
            h[c] += 1## se guarda en el arreglo
    return h## regresa ese arreglo

print("Imagen 2: ", ima.shape)## se imprime los valores
hist2 = histGris( ima )## se llama la funcion
plt.figure(2)## se pone que sea la figura 2
plt.title("Mejorada")## se pone el titulo
plt.subplot(313)
plt.plot(hist2, color = 'r')## que el color se rojo
plt.show() ## se muestra la imagen


print("Imagen 1: ", imagen.shape)## se imprime los valores
hist = histGris( imagen )## se llama la funcion
plt.figure(1)## se pone que sea la figura 1
plt.title("Original")## se pone el titulo de proginal
plt.subplot(312)
plt.plot(hist, color = 'g')## que se muestre de color verde
plt.show() ## se muestra la imagen

