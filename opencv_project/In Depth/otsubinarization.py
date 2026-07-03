import cv2 as cv
import matplotlib.pyplot as plt

def otsuThresholding():
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cat.jpg"

    # Read image in grayscale
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    # Display Original Image
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(imgGray, cmap='gray')
    plt.title("Gray Image")
    plt.axis("off")

    # Global Threshold
    thresh = 70
    maxVal = 255

    _, imgGlobal = cv.threshold(
        imgGray,
        thresh,
        maxVal,
        cv.THRESH_BINARY
    )

    plt.subplot(1, 3, 2)
    plt.imshow(imgGlobal, cmap='gray')
    plt.title("Global Threshold")
    plt.axis("off")

    # Otsu Threshold
    _, imgOtsu = cv.threshold(
        imgGray,
        0,
        maxVal,
        cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    plt.subplot(1, 3, 3)
    plt.imshow(imgOtsu, cmap='gray')
    plt.title("Otsu Threshold")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    otsuThresholding()