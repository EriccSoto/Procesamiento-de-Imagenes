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

ruta = ("Lena.png")
im = cv2.imread( ruta,cv2.IMREAD_GRAYSCALE)
cv2.imshow('Normal', im)
a = [ [ -1.0, -1.0, -1.0 ],
           [ -1.0, 9.0, -1.0 ],
           [ -1.0, -1.0, -1.0 ] ]
kernel = np.asarray(a)
dst = cv2.filter2D(im, -1, kernel)
cv2.imshow('Pasa bajas 1', dst)
cv2.waitKey(0)