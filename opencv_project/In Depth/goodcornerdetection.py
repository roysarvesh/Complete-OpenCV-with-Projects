import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def goodCornerDetection():

    # ---------------------------------------
    # Read Image
    # ---------------------------------------
    imgPath = r"C:\Users\dronr\Documents\opencv_project\coin.jpg"

    img = cv.imread(imgPath)

    if img is None:
        print("Image not found!")
        return

    # ---------------------------------------
    # Resize Image
    # ---------------------------------------
    scale = 40  # Percentage

    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)

    img = cv.resize(img, (width, height))

    # Convert BGR to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Convert to Gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ---------------------------------------
    # Good Features to Track
    # ---------------------------------------
    maxCorners = 200
    qualityLevel = 0.01
    minDistance = 20

    corners = cv.goodFeaturesToTrack(
        gray,
        maxCorners,
        qualityLevel,
        minDistance
    )

    # Draw Corners
    if corners is not None:

        corners = np.int32(corners)

        for corner in corners:

            x, y = corner.ravel()

            cv.circle(
                imgRGB,
                (x, y),
                6,
                (255, 0, 0),
                -1
            )

    # ---------------------------------------
    # Display Results
    # ---------------------------------------
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(imgRGB)
    plt.title("Good Features to Track")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    goodCornerDetection()