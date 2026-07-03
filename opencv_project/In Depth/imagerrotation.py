import cv2 as cv
import matplotlib.pyplot as plt

def rotation():

    # Read image
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"
    img = cv.imread(imgPath)

    if img is None:
        print("Error: Could not load image.")
        return

    # Convert BGR to RGB for Matplotlib
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Get image dimensions
    height, width = imgRGB.shape[:2]

    # Rotation Matrix
    # (Center of rotation, Angle, Scale)
    T = cv.getRotationMatrix2D((width / 2, height / 2), 180, 1)

    # Rotate image
    imgRotate = cv.warpAffine(imgRGB, T, (width, height))

    # Display images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(imgRGB)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(imgRotate)
    plt.title("Rotated Image (180°)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rotation()