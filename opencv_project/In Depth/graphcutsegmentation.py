import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def graphCutSegmentation():

    # ----------------------------------------
    # Read Image
    # ----------------------------------------
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cat.jpg"

    img = cv.imread(imgPath)

    if img is None:
        print("Image not found!")
        return

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # ----------------------------------------
    # Create Mask
    # ----------------------------------------
    mask = np.zeros(img.shape[:2], np.uint8)

    # Background & Foreground Models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # ----------------------------------------
    # Rectangle
    # (x, y, width, height)
    # ----------------------------------------
    rect = (180, 15, 300, 455)

    # ----------------------------------------
    # Apply GrabCut
    # ----------------------------------------
    cv.grabCut(
        img,
        mask,
        rect,
        bgdModel,
        fgdModel,
        5,
        cv.GC_INIT_WITH_RECT
    )

    # ----------------------------------------
    # Create Binary Mask
    # ----------------------------------------
    mask2 = np.where(
        (mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),
        0,
        1
    ).astype("uint8")

    # Segmented Image
    result = imgRGB * mask2[:, :, np.newaxis]

    # White Background (Optional)
    whiteBG = np.ones_like(imgRGB) * 255
    resultWhite = whiteBG.copy()
    resultWhite[mask2 == 1] = imgRGB[mask2 == 1]

    # ----------------------------------------
    # Draw Rectangle for Display
    # ----------------------------------------
    rectangleImage = imgRGB.copy()

    x, y, w, h = rect

    cv.rectangle(
        rectangleImage,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # ----------------------------------------
    # Display
    # ----------------------------------------
    plt.figure(figsize=(18, 6))

    plt.subplot(141)
    plt.imshow(rectangleImage)
    plt.title("Rectangle")
    plt.axis("off")

    plt.subplot(142)
    plt.imshow(mask2, cmap="gray")
    plt.title("Foreground Mask")
    plt.axis("off")

    plt.subplot(143)
    plt.imshow(result)
    plt.title("Black Background")
    plt.axis("off")

    plt.subplot(144)
    plt.imshow(resultWhite)
    plt.title("White Background")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    graphCutSegmentation()