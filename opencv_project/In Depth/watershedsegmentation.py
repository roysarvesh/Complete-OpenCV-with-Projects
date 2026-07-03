import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def watershedSegmentation():

    # ------------------------------------
    # Read Image
    # ------------------------------------
    imgPath = r"C:\Users\dronr\Documents\opencv_project\coin.jpg"

    img = cv.imread(imgPath)

    if img is None:
        print("Image not found!")
        return

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # ------------------------------------
    # Convert to Grayscale
    # ------------------------------------
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ------------------------------------
    # Gaussian Blur
    # ------------------------------------
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    # ------------------------------------
    # Otsu Threshold
    # ------------------------------------
    ret, thresh = cv.threshold(
        blur,
        0,
        255,
        cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )

    # ------------------------------------
    # Morphological Opening
    # ------------------------------------
    kernel = np.ones((3, 3), np.uint8)

    opening = cv.morphologyEx(
        thresh,
        cv.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # ------------------------------------
    # Sure Background
    # ------------------------------------
    sure_bg = cv.dilate(
        opening,
        kernel,
        iterations=3
    )

    # ------------------------------------
    # Distance Transform
    # ------------------------------------
    dist_transform = cv.distanceTransform(
        opening,
        cv.DIST_L2,
        5
    )

    # ------------------------------------
    # Sure Foreground
    # ------------------------------------
    ret, sure_fg = cv.threshold(
        dist_transform,
        0.5 * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    # ------------------------------------
    # Unknown Region
    # ------------------------------------
    unknown = cv.subtract(
        sure_bg,
        sure_fg
    )

    # ------------------------------------
    # Connected Components
    # ------------------------------------
    ret, markers = cv.connectedComponents(
        sure_fg
    )

    markers = markers + 1

    markers[unknown == 255] = 0

    # ------------------------------------
    # Watershed
    # ------------------------------------
    watershedImg = imgRGB.copy()

    markers = cv.watershed(
        img,
        markers
    )

    # Draw Boundary in Red
    watershedImg[markers == -1] = [255, 0, 0]

    # ------------------------------------
    # Display Results
    # ------------------------------------
    plt.figure(figsize=(18, 10))

    plt.subplot(241)
    plt.imshow(imgRGB)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(242)
    plt.imshow(gray, cmap="gray")
    plt.title("Gray")
    plt.axis("off")

    plt.subplot(243)
    plt.imshow(thresh, cmap="gray")
    plt.title("Threshold")
    plt.axis("off")

    plt.subplot(244)
    plt.imshow(opening, cmap="gray")
    plt.title("Opening")
    plt.axis("off")

    plt.subplot(245)
    plt.imshow(sure_bg, cmap="gray")
    plt.title("Sure Background")
    plt.axis("off")

    plt.subplot(246)
    plt.imshow(dist_transform, cmap="jet")
    plt.title("Distance Transform")
    plt.axis("off")

    plt.subplot(247)
    plt.imshow(sure_fg, cmap="gray")
    plt.title("Sure Foreground")
    plt.axis("off")

    plt.subplot(248)
    plt.imshow(watershedImg)
    plt.title("Watershed Result")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    watershedSegmentation()