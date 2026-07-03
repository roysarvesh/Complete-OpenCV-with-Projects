import cv2 as cv
import matplotlib.pyplot as plt

def histogramEqualCLAHE():

    # Read image in grayscale
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Could not load image.")
        return

    # ---------------- Original Histogram ----------------
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdfNorm = cdf * float(hist.max()) / cdf.max()

    # ---------------- Histogram Equalization ----------------
    equImg = cv.equalizeHist(img)

    equHist = cv.calcHist([equImg], [0], None, [256], [0, 256])
    equCdf = equHist.cumsum()
    equCdfNorm = equCdf * float(equHist.max()) / equCdf.max()

    # ---------------- CLAHE ----------------
    clahe = cv.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    claheImg = clahe.apply(img)

    claheHist = cv.calcHist([claheImg], [0], None, [256], [0, 256])
    claheCdf = claheHist.cumsum()
    claheCdfNorm = claheCdf * float(claheHist.max()) / claheCdf.max()

    # ---------------- Display ----------------
    plt.figure(figsize=(15, 10))

    # Original Image
    plt.subplot(3, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    # Original Histogram
    plt.subplot(3, 2, 2)
    plt.plot(hist, color='black')
    plt.plot(cdfNorm, color='blue')
    plt.title("Original Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.xlim([0, 256])

    # Equalized Image
    plt.subplot(3, 2, 3)
    plt.imshow(equImg, cmap='gray')
    plt.title("Histogram Equalization")
    plt.axis("off")

    # Equalized Histogram
    plt.subplot(3, 2, 4)
    plt.plot(equHist, color='black')
    plt.plot(equCdfNorm, color='blue')
    plt.title("Equalized Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.xlim([0, 256])

    # CLAHE Image
    plt.subplot(3, 2, 5)
    plt.imshow(claheImg, cmap='gray')
    plt.title("CLAHE Image")
    plt.axis("off")

    # CLAHE Histogram
    plt.subplot(3, 2, 6)
    plt.plot(claheHist, color='black')
    plt.plot(claheCdfNorm, color='blue')
    plt.title("CLAHE Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    histogramEqualCLAHE()