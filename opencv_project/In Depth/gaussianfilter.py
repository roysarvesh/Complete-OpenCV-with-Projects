import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate Gaussian Kernel
def gaussianKernel(size, sigma):

    kernel = cv.getGaussianKernel(size, sigma)
    kernel = np.outer(kernel, kernel)

    return kernel


def gaussianFiltering():

    # Read image
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cat.jpg"
    img = cv.imread(imgPath)

    if img is None:
        print("Error: Could not load image.")
        return

    # Convert BGR to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Kernel Size and Sigma
    n = 51
    sigma = 8

    # Generate Gaussian Kernel
    kernel = gaussianKernel(n, sigma)

    # Apply Gaussian Blur
    imgFilter = cv.GaussianBlur(imgRGB, (n, n), sigma)

    # Display
    fig = plt.figure(figsize=(15, 5))

    # Gaussian Kernel
    plt.subplot(131)
    plt.imshow(kernel, cmap='viridis')
    plt.title("Gaussian Kernel")
    plt.axis("off")

    # Original Image
    plt.subplot(132)
    plt.imshow(imgRGB)
    plt.title("Original Image")
    plt.axis("off")

    # Filtered Image
    plt.subplot(133)
    plt.imshow(imgFilter)
    plt.title("Gaussian Filtered")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # 3D Surface Plot of Gaussian Kernel
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(0, n, 1)
    y = np.arange(0, n, 1)

    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, kernel, cmap='viridis')

    ax.set_title("3D Gaussian Kernel")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Weight")

    plt.show()


if __name__ == "__main__":
    gaussianFiltering()
"""
import cv2 as cv

def callback(x):
    pass

def gaussianFiltering():

    # Read image
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cat.jpg"
    img = cv.imread(imgPath)

    if img is None:
        print("Error: Could not load image.")
        return

    # Resize image
    height, width = img.shape[:2]
    scale = 0.5

    width = int(width * scale)
    height = int(height * scale)

    img = cv.resize(img, (width, height))

    winName = "Gaussian Filtering"

    cv.namedWindow(winName)

    # Sigma Trackbar
    cv.createTrackbar("Sigma x10", winName, 10, 100, callback)

    kernelSize = 31  # Must be odd

    while True:

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Read sigma from trackbar
        sigma = cv.getTrackbarPos("Sigma x10", winName) / 10.0

        # Sigma cannot be zero
        if sigma == 0:
            sigma = 0.1

        # Apply Gaussian Blur
        gaussianImg = cv.GaussianBlur(
            img,
            (kernelSize, kernelSize),
            sigma
        )

        # Display Sigma value
        display = gaussianImg.copy()
        cv.putText(display,
                   f"Sigma = {sigma:.1f}",
                   (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1,
                   (0, 255, 0),
                   2)

        cv.imshow(winName, display)

    cv.destroyAllWindows()


if __name__ == "__main__":
    gaussianFiltering()"""