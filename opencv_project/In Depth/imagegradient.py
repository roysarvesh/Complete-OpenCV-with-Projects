import cv2 as cv
import matplotlib.pyplot as plt

def imageGradient():

    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"

    # Read image in grayscale
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not found!")
        return

    plt.figure(figsize=(10, 8))

    # ---------------- Original ----------------
    plt.subplot(221)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    # ---------------- Laplacian ----------------
    laplacian = cv.Laplacian(img, cv.CV_64F, ksize=21)

    plt.subplot(222)
    plt.imshow(laplacian, cmap='gray')
    plt.title("Laplacian")
    plt.axis("off")

    # Print Sobel X Kernel
    kx, ky = cv.getDerivKernels(1, 0, 3)
    print("Sobel X Kernel:")
    print(ky @ kx.T)

    # ---------------- Sobel X ----------------
    sobelX = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=21)

    plt.subplot(223)
    plt.imshow(sobelX, cmap='gray')
    plt.title("Sobel X")
    plt.axis("off")

    # Print Sobel Y Kernel
    kx, ky = cv.getDerivKernels(0, 1, 3)
    print("\nSobel Y Kernel:")
    print(ky @ kx.T)

    # ---------------- Sobel Y ----------------
    sobelY = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=21)

    plt.subplot(224)
    plt.imshow(sobelY, cmap='gray')
    plt.title("Sobel Y")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    imageGradient()