import cv2 as cv
import matplotlib.pyplot as plt

def adaptiveThresholding():
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"

    # Read image
    img = cv.imread(imgPath)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Normal Threshold
    _, normalThresh = cv.threshold(
        imgGray, 127, 255, cv.THRESH_BINARY
    )

    # Adaptive Mean Threshold
    meanThresh = cv.adaptiveThreshold(
        imgGray,
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY,
        11,     # Block Size (must be odd)
        2       # Constant C
    )

    # Adaptive Gaussian Threshold
    gaussianThresh = cv.adaptiveThreshold(
        imgGray,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        11,     # Block Size (must be odd)
        2       # Constant C
    )

    # Display Images
    plt.figure(figsize=(10,8))

    plt.subplot(2,2,1)
    plt.imshow(imgGray, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.imshow(normalThresh, cmap="gray")
    plt.title("Normal Threshold")
    plt.axis("off")

    plt.subplot(2,2,3)
    plt.imshow(meanThresh, cmap="gray")
    plt.title("Adaptive Mean")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.imshow(gaussianThresh, cmap="gray")
    plt.title("Adaptive Gaussian")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    adaptiveThresholding()