import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def morphTransformations():
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cat.jpg"
    img = cv.imread(imgPath)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgGaus = cv.GaussianBlur(imgGray, (5, 5), 0)
    kernel = np.ones((7, 7), np.uint8)
    erosion = cv.erode(imgGaus, kernel, iterations=1)
    dilation = cv.dilate(imgGaus, kernel, iterations=1)
    opening = cv.morphologyEx(imgGaus, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(imgGaus, cv.MORPH_CLOSE, kernel)
    gradient = cv.morphologyEx(imgGaus, cv.MORPH_GRADIENT, kernel)
    tophat = cv.morphologyEx(imgGaus, cv.MORPH_TOPHAT, kernel)
    blackhat = cv.morphologyEx(imgGaus, cv.MORPH_BLACKHAT, kernel)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 4, 1)
    plt.imshow(imgGray, cmap='gray')
    plt.title("Original")
    plt.axis("off")
    plt.subplot(2, 4, 2)
    plt.imshow(erosion, cmap='gray')
    plt.title("Erosion")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(dilation, cmap='gray')
    plt.title("Dilation")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(opening, cmap='gray')
    plt.title("Opening")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(closing, cmap='gray')
    plt.title("Closing")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(gradient, cmap='gray')
    plt.title("Gradient")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(tophat, cmap='gray')
    plt.title("Top Hat")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.imshow(blackhat, cmap='gray')
    plt.title("Black Hat")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    morphTransformations()