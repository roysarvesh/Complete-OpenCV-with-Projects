import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def convolution2d():

    # Read image
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"
    img = cv.imread(imgPath)

    if img is None:
        print("Error: Could not load image.")
        return

    # Convert BGR to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Kernel size
    n = 15

    # Average filter kernel
    kernel = np.ones((n, n), np.float32) / (n * n)

    # Apply 2D Convolution
    imgFilter = cv.filter2D(imgRGB, -1, kernel)

    # Display
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(imgRGB)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(imgFilter)
    plt.title(f"{n}x{n} Average Filter")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    convolution2d()