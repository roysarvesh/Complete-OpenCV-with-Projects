import cv2 as cv
import matplotlib.pyplot as plt

def imageResize():
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cat.jpg"

    img = cv.imread(imgPath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    height, width = img.shape[:2]

    scale = 1 / 4

    interpMethods = [
        cv.INTER_AREA,
        cv.INTER_LINEAR,
        cv.INTER_NEAREST,
        cv.INTER_CUBIC,
        cv.INTER_LANCZOS4
    ]

    interpTitle = [
        "Original",
        "INTER_AREA",
        "INTER_LINEAR",
        "INTER_NEAREST",
        "INTER_CUBIC",
        "INTER_LANCZOS4"
    ]

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title(interpTitle[0])
    plt.axis("off")

    for i in range(len(interpMethods)):
        plt.subplot(2, 3, i + 2)

        imgResize = cv.resize(
            img,
            (int(width * scale), int(height * scale)),
            interpolation=interpMethods[i]
        )

        plt.imshow(imgResize)
        plt.title(interpTitle[i + 1])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    imageResize()