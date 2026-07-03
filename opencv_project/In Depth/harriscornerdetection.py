import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def harrisCorner():

    # ----------------------------------------
    # Read Image
    # ----------------------------------------
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cat.jpg"

    img = cv.imread(imgPath)

    if img is None:
        print("Image not found!")
        return

    # Convert BGR to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Convert to Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Convert to float32 (Required)
    gray = np.float32(gray)

    # ----------------------------------------
    # Harris Corner Detection
    # ----------------------------------------
    blockSize = 5
    sobelSize = 3
    k = 0.04

    harris = cv.cornerHarris(
        gray,
        blockSize,
        sobelSize,
        k
    )

    # Dilate corner points
    harris = cv.dilate(harris, None)

    # Copy original image
    result = imgRGB.copy()

    # Mark detected corners in Red
    result[harris > 0.05 * harris.max()] = [255, 0, 0]

    # ----------------------------------------
    # Display Results
    # ----------------------------------------
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(imgRGB)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(132)
    plt.imshow(harris, cmap="gray")
    plt.title("Harris Response")
    plt.axis("off")

    plt.subplot(133)
    plt.imshow(result)
    plt.title("Detected Corners")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    harrisCorner()