import cv2
import numpy as np
import math
from math import cos, radians

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
        cv2.imshow("R",backR))
        cv2.imshow("g",backG)
        cv2.imshow("B",backB)
        cv2.waitKey(0)
        #final = final/360*255

        # Convert from flot32 to uint8:
        final = np.round(final * 255).astype(np.uint8)
        return final


##### Converting to HSI using OpenCV is simple, converting back is difficult #####
def bgr2hsi(bgr):
    """Convert image from BGR color format to HSI color format"""
    # Convert from BGR to HSV using OpenCV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Difference between HSV and HSI:
    # In HSV: V = max(R, G, B)
    # In HSI: I = (R + G + B)/3
    hsi = hsv
    hsi[:, :, 2] = np.mean(bgr, 2)

    return hsi


bgr = cv2.imread('Lena.png')

#hsi = bgr2hsi(bgr)

average = np.mean(np.float32(bgr)/255, 2)
new_bgr = HSI_TO_RGB(average, bgr)

cv2.imshow('new_bgr', new_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()