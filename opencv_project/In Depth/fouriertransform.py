import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def fourierTransform():

    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"

    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not found!")
        return

    plt.figure(figsize=(14,8))

    # ---------------- Original ----------------
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    # ---------------- Fourier Transform ----------------
    imgDFT = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)

    imgDFT_DB = 20 * np.log(
        cv.magnitude(imgDFT[:, :, 0], imgDFT[:, :, 1]) + 1
    )

    plt.subplot(232)
    plt.imshow(imgDFT_DB, cmap='gray')
    plt.title("Fourier Spectrum")
    plt.axis("off")

    # ---------------- Shift ----------------
    imgDFTShift = np.fft.fftshift(imgDFT)

    imgDFTShift_DB = 20 * np.log(
        cv.magnitude(
            imgDFTShift[:, :, 0],
            imgDFTShift[:, :, 1]
        ) + 1
    )

    plt.subplot(233)
    plt.imshow(imgDFTShift_DB, cmap='gray')
    plt.title("Shifted Spectrum")
    plt.axis("off")

    # ---------------- Low Pass Mask ----------------
    rows, cols = img.shape

    mask = np.zeros((rows, cols, 2), np.uint8)

    offset = 50

    mask[
        rows//2-offset:rows//2+offset,
        cols//2-offset:cols//2+offset
    ] = 1

    plt.subplot(234)
    plt.imshow(mask[:, :, 0], cmap='gray')
    plt.title("Low Pass Mask")
    plt.axis("off")

    # ---------------- Apply Mask ----------------
    imgDFTShift_LP = imgDFTShift * mask

    imgDFTShift_LP_DB = 20 * np.log(
        cv.magnitude(
            imgDFTShift_LP[:, :, 0],
            imgDFTShift_LP[:, :, 1]
        ) + 1
    )

    plt.subplot(235)
    plt.imshow(imgDFTShift_LP_DB, cmap='gray')
    plt.title("Low Pass Spectrum")
    plt.axis("off")

    # ---------------- Inverse DFT ----------------
    imgInvDFT = np.fft.ifftshift(imgDFTShift_LP)

    imgIDFT = cv.idft(imgInvDFT)

    imgLP = cv.magnitude(
        imgIDFT[:, :, 0],
        imgIDFT[:, :, 1]
    )

    plt.subplot(236)
    plt.imshow(imgLP, cmap='gray')
    plt.title("Low Pass Output")
    plt.axis("off")

    # ==================================================
    # Fourier Transform of Gaussian & Laplacian Kernels
    # ==================================================

    coef = cv.getGaussianKernel(7, 5)
    gaussianKernel = coef @ coef.T

    laplacianKernel = np.array([
        [0, 1, 0],
        [1,-4, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    plt.figure(figsize=(10,5))

    # Gaussian Kernel FFT
    gaussFFT = np.fft.fft2(gaussianKernel)

    gaussFFTShift = np.fft.fftshift(gaussFFT)

    gaussMag = np.log(np.abs(gaussFFTShift)+1)

    plt.subplot(121)
    plt.imshow(gaussMag, cmap='gray')
    plt.title("Gaussian Kernel FFT")
    plt.axis("off")

    # Laplacian Kernel FFT
    lapFFT = np.fft.fft2(laplacianKernel)

    lapFFTShift = np.fft.fftshift(lapFFT)

    lapMag = np.log(np.abs(lapFFTShift)+1)

    plt.subplot(122)
    plt.imshow(lapMag, cmap='gray')
    plt.title("Laplacian Kernel FFT")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fourierTransform()