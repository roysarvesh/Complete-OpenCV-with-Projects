import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def imageTranslation():

    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"
    img = cv.imread(imgPath)

    if img is None:
        print("Error: Could not load image.")
        return

    # Convert BGR to RGB for Matplotlib
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Translation Matrix
    # Move 150 pixels right and 100 pixels down
    T = np.array([
        [1, 0, 150],
        [0, 1, 100]
    ], dtype=np.float32)

    # Get image dimensions
    height, width = img.shape[:2]

    # Apply Translation
    translated = cv.warpAffine(imgRGB, T, (width, height))

    # Display Images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(imgRGB)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(translated)
    plt.title("Translated Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    imageTranslation()