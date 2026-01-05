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
import matplotlib.image as mpimg
import pandas as pd
from math import cos, radians
##_________________
rgb_scale = 255
cmyk_scale = 100
def r_g_b():
    img = cv2.imread('B.png')
    b, g ,r =cv2.split(img)

    zeros = np.zeros(img.shape[:2], dtype = "uint8")          
    merged_r = cv2.merge ([zeros, zeros, r]) #El componente del canal es cero se puede entender como una matriz cero
    merged_g = cv2.merge ([zeros, g, zeros])
    merged_b = cv2.merge ([b, zeros, zeros])
    cv2.imshow('image',img)
    cv2.imshow("Rojo",merged_r)
    cv2.imshow("Verde",merged_g)
    cv2.imshow("Azul",merged_b)
    cv2.waitKey(0)
def ycbcr():
    ima = mpimg.imread('Lena.png')
    ima_r = ima[:, :, 0]
    ima_g = ima[:, :, 1]
    ima_b = ima[:, :, 2]

#Obtener brillo, que es una copia en escala de grises de la imagen original
    ima_y = 0.256789 * ima_r + 0.504129 * ima_g + 0.097906 * ima_b + 16
 #Obtener el componente azul
    ima_cb = -0.148223 * ima_r - 0.290992 * ima_g + 0.439215 * ima_b + 128
 #Obtener el componente rojo
    ima_cr = 0.439215 * ima_r - 0.367789 * ima_g - 0.071426 * ima_b + 128
 
# Combina los tres componentes juntos
    ima_rgb2ycbcr = np.zeros(ima.shape)
    ima_rgb2ycbcr[:,:,0] = ima_y
    ima_rgb2ycbcr[:,:,1] = ima_cb
    ima_rgb2ycbcr[:,:,2] = ima_cr

    plt.imshow(ima_y)
    plt.title("Y")
    plt.show()
    plt.imshow(ima_cb)
    plt.title("CB") 
    plt.show()
    plt.imshow(ima_cr)
    plt.title("CR") 
    plt.show()

    ima_ycbcr2rgb = np.zeros(ima.shape)
    ima_ycbcr2rgb[:,:,0] = 1.164383 * (ima_y-16) + 1.596027 * (ima_cr-128)
    ima_ycbcr2rgb[:,:,1] = 1.164383 * (ima_y-16) - 0.391762 * (ima_cb-128)- 0.812969 * (ima_cr-128)
    ima_ycbcr2rgb[:,:,2] = 1.164383 * (ima_y-16) + 2.017230 * (ima_cb-128)

    plt.imshow(ima_ycbcr2rgb[:,:,0])
    plt.title("R")
    plt.show()
    plt.imshow(ima_ycbcr2rgb[:,:,1])
    plt.title("G")
    plt.show()
    plt.imshow(ima_ycbcr2rgb[:,:,2])
    plt.title("B")
    plt.show()
def rgb2ycbcr():
    ima = mpimg.imread('Lena.png')
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = ima.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    plt.imshow(ycbcr[:,:,0])
    plt.title("y")
    plt.show()
    plt.imshow(ycbcr[:,:,1])
    plt.title("cb")
    plt.show()
    plt.imshow(ycbcr[:,:,2])
    plt.title("cr")
    plt.show()
def ycbcr2rgb():
    ima = mpimg.imread('Lena.png')
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = ima.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    plt.imshow(rgb[:,:,0])
    plt.title("r")
    plt.show()
    plt.imshow(rgb[:,:,1])
    plt.title("g")
    plt.show()
    plt.imshow(rgb[:,:,2])
    plt.title("b")
    plt.show()

def rgb_to_cmyk():
    img = cv2.imread('Lena.png')
    b, g ,r =cv2.split(img)

    zeros = np.zeros(img.shape[:2], dtype = "uint8")          
    merged_r = cv2.merge ([zeros, zeros, r]) #El componente del canal es cero se puede entender como una matriz cero
    merged_g = cv2.merge ([zeros, g, zeros])
    merged_b = cv2.merge ([b, zeros, zeros])
    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0, 0, 0, cmyk_scale

    # rgb [0,255] -> cmy [0,1]
    c = 1 - merged_r / float(rgb_scale)
    m = 1 - merged_g / float(rgb_scale)
    y = 1 - merged_b / float(rgb_scale)

    # extract out k [0,1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) 
    m = (m - min_cmy) 
    y = (y - min_cmy) 
    k = min_cmy
    c1 = c*cmyk_scale
    m1 = m*cmyk_scale
    y1 = y*cmyk_scale
    k1= k*cmyk_scale

    cv2.imshow("c",c1)
    cv2.imshow("m",m1)
    cv2.imshow("y",y1)
    cv2.imshow("k",k1)
    cv2.waitKey(0)

def rgb_cmyk():
    # Import image
    img = plt.imread("ima1.jpg")

# Create float
    bgr = img.astype(float)/255.

# Extract channels
    with np.errstate(invalid='ignore', divide='ignore'):
        K = 1 - np.max(bgr, axis=2)
        C = (1-bgr[...,2] - K)/(1-K)
        M = (1-bgr[...,1] - K)/(1-K)
        Y = (1-bgr[...,0] - K)/(1-K)

# Convert the input BGR image to CMYK colorspace
    CMYK = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)

# Split CMYK channels
    Y, M, C, K = cv2.split(CMYK)

    np.isfinite(C).all()
    np.isfinite(M).all()
    np.isfinite(K).all()
    np.isfinite(Y).all()

    cv2.imshow("c",C)
    cv2.imshow("m",M)
    cv2.imshow("y",Y)
    cv2.imshow("k",K)
    

    cmyk_to_rgb(C,M,Y,K)

def cmyk_to_rgb(c,m,y,k):
    r = rgb_scale*(1.0-(c+k)/float(cmyk_scale))
    g = rgb_scale*(1.0-(m+k)/float(cmyk_scale))
    b = rgb_scale*(1.0-(y+k)/float(cmyk_scale))
    cv2.imshow("R",r)
    cv2.imshow("G",g)
    cv2.imshow("B",b)
    cv2.waitKey(0)

def HSI_to_bgr(h, s, i):
    h = np.degrees(h)
    if 0 < h <= 512 :
        b = i * (1 - s)
        r = i * (1 + (s * np.cos(np.radians(h)) / np.cos(np.radians(60) - np.radians(h))))
        g = i * 3 - (r + b)
    elif 120 < h <= 240:
        h -= 120
        r = i * (1 - s)
        g = i * (1 + (s * np.cos(np.radians(h)) / np.cos(np.radians(60) - np.radians(h))))
        b = 3 * i - (r + g)
    elif 0 < h <= 360:
        h -= 240
        g = i * (1 - s)
        b = i * (1 + (s * np.cos(np.radians(h)) / np.cos(np.radians(60) - np.radians(h))))
        r = i * 3 - (g + b)
    rgb= [b, g, r]
    cv2.imshow("rgb",rgb)
    cv2.imshow("r",r)
    cv2.imshow("g",g)
    cv2.imshow("b",b)
    cv2.waitKey(0)


def RGB_TO_HSI(img):

    with np.errstate(divide='ignore', invalid='ignore'):

        #Load image with 32 bit floats as variable type
        bgr = np.float32(img)/255

        #Separate color channels
        blue = bgr[:,:,0]
        green = bgr[:,:,1]
        red = bgr[:,:,2]

        #Calculate Intensity
        def calc_intensity(red, blue, green):
            return np.divide(blue + green + red, 3)

        #Calculate Saturation
        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

            return saturation

        #Calculate Hue
        def calc_hue(red, blue, green):
            hue = np.copy(red)

            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                math.sqrt((red[i][j] - green[i][j])**2 +
                                        ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])

                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

            return hue

        #Merge channels into picture and return image
        hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green)))
        cv2.imshow("HSI",hsi)
        data =  pd.DataFrame(calc_hue(red, blue, green))
        print(data)
        cv2.imshow("H",calc_hue(red, blue, green))
        cv2.imshow("S",calc_saturation(red, blue, green))
        cv2.imshow("I",calc_intensity(red, blue, green))
        cv2.waitKey(0)
       # HSI_to_bgr(calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green))
        

def HSI_TO_RGB(Average, img):
     with np.errstate(divide='ignore', invalid='ignore'):

        #Load image with 32 bit floats as variable type
        bgr = np.float32(img)/255

        #Separate color channels
        blue = bgr[:,:,0]
        green = bgr[:,:,1]
        red = bgr[:,:,2]

        #Calculate Saturation
        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

            return saturation

        #Calculate Hue
        def calc_hue(red, blue, green):
            hue = np.copy(red)

            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                math.sqrt((red[i][j] - green[i][j])**2 +
                                        ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])

                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

            return hue

        huehue = calc_hue(red, blue, green) # * 255
        satsat = calc_saturation(red, blue, green) # * 255

        huehue = np.nan_to_num(huehue)  # Replace nan with zeros

        # Convert from radians to degrees
        huehue_deg = np.rad2deg(huehue)

        # Initiazlie with zeros
        backR = np.zeros_like(satsat)
        backG = np.zeros_like(satsat)
        backB = np.zeros_like(satsat)

        for i in range(0, satsat.shape[0]):
            for j in range(0, satsat.shape[1]):

                if 0 <= huehue_deg[i][j] < 1:
                   backR[i][j] =  (Average[i][j] + (2 * Average[i][j] * satsat[i][j]))
                   backG[i][j] =  (Average[i][j] - (Average[i][j] * satsat[i][j]))
                   backB[i][j] =  (Average[i][j] - (Average[i][j] * satsat[i][j]))                   

                elif 1 <= huehue_deg[i][j] < 120:
                   backR[i][j] = (Average[i][j] + (Average[i][j] * satsat[i][j]) * cos(huehue[i][j]) / cos(radians(60)-huehue[i][j]))
                   backG[i][j] = (Average[i][j] + (Average[i][j] * satsat[i][j]) * (1 - cos(huehue[i][j]) / cos(radians(60)-huehue[i][j])))
                   backB[i][j] = (Average[i][j] - (Average[i][j] * satsat[i][j]))

                elif 120 <= huehue_deg[i][j] < 121:
                   backR[i][j] = (Average[i][j] - (Average[i][j] * satsat[i][j]))
                   backG[i][j] = (Average[i][j] + (2 * Average[i][j] * satsat[i][j]))
                   backB[i][j] = (Average[i][j] - (Average[i][j] * satsat[i][j]))

                elif 121 <= huehue_deg[i][j] < 240:
                   backR[i][j] = (Average[i][j] - (Average[i][j] * satsat[i][j]))
                   backG[i][j] = (Average[i][j] + (Average[i][j] * satsat[i][j]) * cos(huehue[i][j]-radians(120)) / cos(radians(180)-huehue[i][j]))
                   backB[i][j] = (Average[i][j] + (Average[i][j] * satsat[i][j]) * (1 - cos(huehue[i][j]-radians(120)) / cos(radians(180)-huehue[i][j])))

                elif 240 <= huehue_deg[i][j] < 241:
                   backR[i][j] = (Average[i][j] - (Average[i][j] * satsat[i][j]))
                   backG[i][j] = (Average[i][j] - (Average[i][j] * satsat[i][j]))
                   backB[i][j] = (Average[i][j] + (2 * Average[i][j] * satsat[i][j]))

                else: #elif 241 <= huehue_deg[i][j] < 360:
                   backR[i][j] = (Average[i][j] + (Average[i][j] * satsat[i][j]) * (1 - cos(huehue[i][j]-radians(240)) / cos(radians(300)-huehue[i][j])))
                   backG[i][j] = (Average[i][j] - (Average[i][j] * satsat[i][j]))
                   backB[i][j] = (Average[i][j] + (Average[i][j] * satsat[i][j]) * cos(huehue[i][j]-radians(240)) / cos(radians(300)-huehue[i][j]))

        #final = cv2.merge((backR, backG, backB))

        # The correct order is BGR and not RGB (at the beginning of the funtion: blue = bgr[:,:,0])
        final = cv2.merge((backB, backG, backR))
        cv2.imshow("R",backR)
        cv2.imshow("g",backG)
        cv2.imshow("B",backB)
        cv2.waitKey(0)
        #final = final/360*255

        # Convert from flot32 to uint8:
        final = np.round(final * 255).astype(np.uint8)
        return final


os.system('clear') # NOTA para windows tienes que cambiar clear por cls

print ("Selecciona una opción")
print ("\t1 - RGB")
print ("\t2 - YCBCR")
print ("\t3 - YCBCR(version 2)")
print ("\t4 - CMYK")
print ("\t5 - HSI")
print ("\t5 - Salir")



while True:
    opcionMenu = input("inserta la opción >> ")
    
    if opcionMenu=="1":
        r_g_b()
        
    if opcionMenu=="2":
        ycbcr()
        
    if opcionMenu=="3":
        rgb2ycbcr()
        ycbcr2rgb()
        
    if opcionMenu=="4":
        rgb_cmyk()
        
    if opcionMenu=="5":
        img = cv2.imread('Lena.png')
        RGB_TO_HSI(img)
        average = np.mean(np.float32(img)/255, 2)
        new_bgr = HSI_TO_RGB(average, img)
        cv2.imshow('Nueva imagen', new_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if opcionMenu=="5":  
        break
    else:
        input("No has pulsado ninguna opción correcta...\npulsa una tecla para continuar")