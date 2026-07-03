from logging import root

import cv2 as cv
import os
import matplotlib.pyplot as plt


def readAndWriteSinglePixel():
    root = os.getcwd()
    imgPath = os.path.join(root, 'cat.jpg')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()
    eyePixel = imgRGB[200,450]
    imgRGB[200,450] = [255, 0, 0]
    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

def readAndWritePixelRegion():
    root = os.getcwd()
    imgPath = os.path.join(root, 'cat.jpg')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(imgRGB)
    plt.show()
    eyeRegion = imgRGB[185:255, 380:445]
    dx = 70
    dy = 65
    startX = 60
    startY = 70
    imgRGB[startX:startX + dx, startY:startY + dy] = eyeRegion
    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

if __name__ == '__main__':
    readAndWriteSinglePixel()
    readAndWritePixelRegion()