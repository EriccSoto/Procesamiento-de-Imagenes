from __future__ import division
import math
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np

img1 = cv2.imread('ima1.jpg')
img2 = cv2.imread('ima2.jpg')
resA = cv2.add(img1,img2)
cv2.imshow('resA',resA)
cv2.waitKey(0)
cv2.destroyAllWindows()