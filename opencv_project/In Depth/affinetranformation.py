import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def affineTransform():

    # Read image
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"
    img = cv.imread(imgPath)

    if img is None:
        print("Error: Could not load image.")
        return

    # Convert BGR to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    height, width = imgRGB.shape[:2]

    # Three points from the original image
    pts1 = np.float32([
        [50, 50],
        [200, 50],
        [50, 200]
    ])

    # Corresponding points in the transformed image
    pts2 = np.float32([
        [10, 100],
        [200, 50],
        [100, 250]
    ])

    # Compute affine transformation matrix
    M = cv.getAffineTransform(pts1, pts2)

    # Apply affine transformation
    affine = cv.warpAffine(imgRGB, M, (width, height))

    # Display images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(imgRGB)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(affine)
    plt.title("Affine Transform")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    affineTransform()