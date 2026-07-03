import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def medianFiltering():

    # Read image
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"
    img = cv.imread(imgPath)

    if img is None:
        print("Error: Could not load image.")
        return

    # Convert BGR to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Add Salt-and-Pepper Noise
    noisyImg = imgRGB.copy()

    noiseProb = 0.05

    noise = np.random.rand(noisyImg.shape[0], noisyImg.shape[1])

    noisyImg[noise < noiseProb / 2] = 0
    noisyImg[noise > 1 - noiseProb / 2] = 255

    # Apply Median Filter
    imgFilter = cv.medianBlur(noisyImg, 5)

    # Display Images
    plt.figure(figsize=(10,5))

    plt.subplot(121)
    plt.imshow(noisyImg)
    plt.title("Noisy Image")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(imgFilter)
    plt.title("Median Filtered Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    medianFiltering()