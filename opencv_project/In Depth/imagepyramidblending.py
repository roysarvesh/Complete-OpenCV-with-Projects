import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def imagePyramidBlending():

    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"

    imgBGR = cv.imread(imgPath)

    if imgBGR is None:
        print("Image not found!")
        return

    imgRGB = cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB)

    # Create second image by flipping the first one
    img2 = cv.flip(imgRGB, 1)

    # -------------------------
    # Gaussian Pyramid - Image 1
    # -------------------------
    gp1 = [imgRGB]
    down = imgRGB.copy()

    for i in range(5):
        down = cv.pyrDown(down)
        gp1.append(down)

    # -------------------------
    # Gaussian Pyramid - Image 2
    # -------------------------
    gp2 = [img2]
    down = img2.copy()

    for i in range(5):
        down = cv.pyrDown(down)
        gp2.append(down)

    # -------------------------
    # Laplacian Pyramid - Image 1
    # -------------------------
    lp1 = [gp1[-1]]

    for i in range(5, 0, -1):

        up = cv.pyrUp(gp1[i])

        if up.shape != gp1[i-1].shape:
            up = cv.resize(up, (gp1[i-1].shape[1], gp1[i-1].shape[0]))

        lap = cv.subtract(gp1[i-1], up)
        lp1.append(lap)

    # -------------------------
    # Laplacian Pyramid - Image 2
    # -------------------------
    lp2 = [gp2[-1]]

    for i in range(5, 0, -1):

        up = cv.pyrUp(gp2[i])

        if up.shape != gp2[i-1].shape:
            up = cv.resize(up, (gp2[i-1].shape[1], gp2[i-1].shape[0]))

        lap = cv.subtract(gp2[i-1], up)
        lp2.append(lap)

    # -------------------------
    # Blend Laplacian Levels
    # -------------------------
    LS = []

    for l1, l2 in zip(lp1, lp2):

        rows, cols, ch = l1.shape

        blend = np.hstack((l1[:, :cols//2],
                           l2[:, cols//2:]))

        LS.append(blend)

    # -------------------------
    # Reconstruct Image
    # -------------------------
    blend = LS[0]

    for i in range(1, len(LS)):

        blend = cv.pyrUp(blend)

        if blend.shape != LS[i].shape:
            blend = cv.resize(blend,
                              (LS[i].shape[1],
                               LS[i].shape[0]))

        blend = cv.add(blend, LS[i])

    # -------------------------
    # Direct Blend
    # -------------------------
    rows, cols, ch = imgRGB.shape

    direct = np.hstack((imgRGB[:, :cols//2],
                        img2[:, cols//2:]))

    # -------------------------
    # Display
    # -------------------------
    plt.figure(figsize=(15,8))

    plt.subplot(131)
    plt.imshow(imgRGB)
    plt.title("Image 1")
    plt.axis("off")

    plt.subplot(132)
    plt.imshow(direct)
    plt.title("Direct Blend")
    plt.axis("off")

    plt.subplot(133)
    plt.imshow(blend)
    plt.title("Pyramid Blend")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    imagePyramidBlending()