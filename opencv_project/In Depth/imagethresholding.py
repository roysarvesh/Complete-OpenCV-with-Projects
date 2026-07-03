import cv2 as cv
import matplotlib.pyplot as plt

def thresholding():
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"

    img = cv.imread(imgPath)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Histogram
    hist = cv.calcHist([imgGray], [0], None, [256], [0, 256])

    plt.figure()
    plt.plot(hist)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    thresOpt = [
        cv.THRESH_BINARY,
        cv.THRESH_BINARY_INV,
        cv.THRESH_TOZERO,
        cv.THRESH_TOZERO_INV,
        cv.THRESH_TRUNC
    ]

    thresNames = [
        "Binary",
        "Binary Inv",
        "To Zero",
        "To Zero Inv",
        "Trunc"
    ]

    plt.figure(figsize=(10, 6))

    # Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(imgGray, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Thresholded Images
    for i in range(len(thresOpt)):
        plt.subplot(2, 3, i + 2)
        _, imgThres = cv.threshold(imgGray, 70, 255, thresOpt[i])
        plt.imshow(imgThres, cmap="gray")
        plt.title(thresNames[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    thresholding()