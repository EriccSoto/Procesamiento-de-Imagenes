import numpy as np
from PIL import Image

import matplotlib.pyplot as plt 

gray_img = Image.open('ima1.jpg').convert("LA")
gray_img.show()
row = gray_img.size[0]
col = gray_img.size[1]
stretch_img = Image.new("L", (row, col))
high = 0
low = 255
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
        stretch_img.putpixel((x,y), int((((gray_img.getpixel((x,y))[0]-low)/(high-low))*(MAX-MIN))+MIN))
stretch_img.show()

y = stretch_img.histogram()
x = np.arange(len(y))
plt.title("Expansion del Histograma")
plt.bar(x, y)
plt.show()