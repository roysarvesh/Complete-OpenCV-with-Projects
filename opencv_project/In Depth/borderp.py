import cv2 as cv
import matplotlib.pyplot as plt

def paddingDemo():

    # Read Image
    imgPath = r"C:\Users\dronr\Documents\opencv_project\lady.jpg"
    img = cv.imread(imgPath)

    if img is None:
        print("Error: Could not load image.")
        return

    # Convert BGR to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    pad = 200

    borderTypes = [
        cv.BORDER_CONSTANT,
        cv.BORDER_REFLECT,
        cv.BORDER_REPLICATE,
        cv.BORDER_WRAP
    ]

    borderTitles = [
        "constant",
        "reflect",
        "replicate",
        "wrap"
    ]

    plt.figure(figsize=(10,8))

    # Original Image
    plt.subplot(231)
    plt.imshow(imgRGB)
    plt.title("original")
    plt.axis("off")

    # Display all border types
    for i in range(len(borderTypes)):

        plt.subplot(2, 3, i + 2)

        imgPad = cv.copyMakeBorder(
            imgRGB,
            pad,
            pad,
            pad,
            pad,
            borderTypes[i]
        )

        plt.imshow(imgPad)
        plt.title(borderTitles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    paddingDemo()