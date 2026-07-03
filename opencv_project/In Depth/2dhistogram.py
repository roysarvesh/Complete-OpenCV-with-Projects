import cv2 as cv
import matplotlib.pyplot as plt

def histogram2D():

    # Read image
    imgPath = r"C:\Users\dronr\Documents\opencv_project\man.jpeg"
    img = cv.imread(imgPath)

    if img is None:
        print("Error: Could not load image.")
        return

    # Convert BGR to RGB for displaying
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Calculate 2D Histogram (Red vs Green Channels)
    hist = cv.calcHist(
        [imgRGB],      # Image
        [0, 1],        # Channels (Red, Green)
        None,          # No mask
        [256, 256],    # Number of bins
        [0, 256, 0, 256]
    )

    # Display
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(imgRGB)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(hist, interpolation='nearest', cmap='hot')
    plt.title("2D Histogram (Red vs Green)")
    plt.xlabel("Green Intensity")
    plt.ylabel("Red Intensity")
    plt.colorbar(label="Pixel Count")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    histogram2D()