import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def perspectiveTransform():

    # Read image
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"
    img = cv.imread(imgPath)

    if img is None:
        print("Error: Could not load image.")
        return

    # Convert BGR to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    height, width = imgRGB.shape[:2]

    # Four source points
    pts1 = np.float32([
        [50, 50],                      # Top-left
        [width - 50, 50],              # Top-right
        [50, height - 50],             # Bottom-left
        [width - 50, height - 50]      # Bottom-right
    ])

    # Four destination points
    pts2 = np.float32([
        [0, 0],
        [width - 100, 50],
        [100, height - 50],
        [width - 50, height - 100]
    ])

    # Compute Perspective Transformation Matrix
    M = cv.getPerspectiveTransform(pts1, pts2)

    # Apply Perspective Transformation
    perspective = cv.warpPerspective(imgRGB, M, (width, height))

    # Display Images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(imgRGB)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(perspective)
    plt.title("Perspective Transform")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    perspectiveTransform()