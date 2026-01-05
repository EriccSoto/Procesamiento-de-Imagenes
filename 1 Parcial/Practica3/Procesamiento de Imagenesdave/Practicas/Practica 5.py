import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from skimage import feature, filters

def His():
  img = cv2.imread('Lena.png',0)
  
  ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
  # Otsu's thresholding
  ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  # Otsu's thresholding after Gaussian filtering
  blur = cv2.GaussianBlur(img,(5,5),0)
  ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  # plot all the images and their histograms
  images = [img, img, blur]

  miArray = np.arange(3)
  plt.title('Origial')
  plt.hist(images[0].ravel(),256)
  plt.show()
  plt.title('Primera version')
  plt.hist(images[1].ravel(),256)
  plt.show()
  plt.title('Segunda version')
  plt.hist(images[2].ravel(),256)
  plt.show()
  plt.title('Orignal')
  plt.imshow(img,'gray')
  plt.show()
  plt.title('Primera funcion Otsu')
  plt.imshow(th2,'gray')
  plt.show()
  plt.title('Segunda funcion Otsu')
  plt.imshow(th3,'gray')
  plt.show()

def his2():
  
  img = cv2.imread('Lena.png',0)

# global thresholding
  ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
  ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
  blur = cv2.GaussianBlur(img,(5,5),0)
  ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
  images = [img, 0, th1, img, 0, th2, blur, 0, th3]
  titles = ['Imgaen Original','Histogram','Global',
  'Imgaen Original','Histogram',"Otsu's (1)",
  'Filtro gaussiano','Histogram',"Otsu's (2)"]
  miArray = np.arange(3)
  for i in miArray:
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
  plt.show()


def bor():
  
 
  img = cv2.imread('su.jpg',0)
 # img = filters.sobel(imag) # Sobel
  # global thresholding
  r1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
  # Otsu's thresholding
  r2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  # Otsu's thresholding after Gaussian filtering
  blur = cv2.GaussianBlur(img,(5,5),0)
  r3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
  
  plt.title('Orignal')
  plt.imshow(img,'gray')
  plt.show()
  plt.title('General')
  plt.imshow(th1,'gray')
  plt.show()
  plt.title('Primera funcion Otsu')
  plt.imshow(th2,'gray')
  plt.show()
  plt.title('Segunda funcion Otsu')
  plt.imshow(th3,'gray')
  plt.show()


def bor2():
    img = cv2.imread('Lena.png', 0)
    img2= cv2.imread('su.jpg', 0)
    edges_sobel = filters.sobel(img) # Sobel
    edges_sobel2 = filters.sobel(img2) # Sobel
    cv2.imshow('Lena',edges_sobel )
    cv2.imshow('Sudoku',edges_sobel2 )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def tran():
  img = cv2.imread('su.jpg')
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

  lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

  for line in lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 1, cv2.LINE_AA)

  cv2.imshow('Transfomada', edges)
  cv2.imshow('Detector de Lineas', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def tran2():
  img = cv2.imread('su.jpg')
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

  lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

  for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img, (x1,y1), (x2,y2), (0, 0, 255), 1, cv2.LINE_AA)

  cv2.imshow('Transfomada', edges)
  cv2.imshow('Detector de Lineas', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

os.system('clear') # NOTA para windows tienes que cambiar clear por cls

print ("Selecciona una opción")
print ("\t1 - Por Histograma")
print ("\t2 - Bordes (1)")
print ("\t3 - Bordes (2)")
print ("\t4 - Hough (1)")
print ("\t5 - Hough (2)")
print ("\t6 - Histograma (2)")
print ("\t7 - Salir")
while True:
    opcionMenu = input("inserta la opción >> ")
    
    if opcionMenu=="1":
      His()
    if opcionMenu=="2":
      bor()
    if opcionMenu=="3":
      bor2()
    if opcionMenu=="4":
      tran()
    if opcionMenu=="5":
      tran2()
    if opcionMenu=="6":
      his2()
    if opcionMenu=="7":  
        break
    else:
        input("No has pulsado ninguna opción correcta...\npulsa una tecla para continuar")