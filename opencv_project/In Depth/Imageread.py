import cv2 as cv
import os
def readImage():
    img = cv.imread("image1.jpg",-1)

    cv.imshow("Input Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def writeImage():
    img = cv.imread("image1.jpg", cv.IMREAD_GRAYSCALE)

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    output_path = os.path.join(desktop, "output.jpg")

    success = cv.imwrite(output_path, img)

    if success:
        print("Image saved successfully!")
        print("Saved at:", output_path)
    else:
        print("Failed to save image!")

if __name__ == "__main__":
    readImage()
    writeImage()
