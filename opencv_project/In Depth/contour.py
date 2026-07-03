import cv2 as cv
import matplotlib.pyplot as plt

def contoursDemo():

    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"

    # Read image
    img = cv.imread(imgPath)

    if img is None:
        print("Image not found!")
        return

    # Convert to RGB for matplotlib
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Binary Threshold
    _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv.findContours(
        thresh,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE
    )

    print("Number of Contours Found :", len(contours))

    # Draw all contours
    contourImage = imgRGB.copy()

    cv.drawContours(
        contourImage,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    # Display Images
    plt.figure(figsize=(12,8))

    plt.subplot(221)
    plt.imshow(imgRGB)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(222)
    plt.imshow(gray, cmap="gray")
    plt.title("Gray Image")
    plt.axis("off")

    plt.subplot(223)
    plt.imshow(thresh, cmap="gray")
    plt.title("Threshold Image")
    plt.axis("off")

    plt.subplot(224)
    plt.imshow(contourImage)
    plt.title("Contours")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    contoursDemo()