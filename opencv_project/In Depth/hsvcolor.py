import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def hsvColorSegmentation():
    img = cv.imread(r'C:\Users\dronr\Documents\opencv_project\cats.jpg')

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lowerBound = np.array([0, 0, 50])
    upperBound = np.array([10, 120, 50])

    mask = cv.inRange(hsv, lowerBound, upperBound)

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

    cv.imshow('Mask', mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

    debug = 1


if __name__ == '__main__':
    hsvColorSegmentation()