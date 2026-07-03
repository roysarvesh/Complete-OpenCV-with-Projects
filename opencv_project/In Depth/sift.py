import cv2 as cv
import matplotlib.pyplot as plt


def siftDetection():

    # ---------------------------------------
    # Read Image
    # ---------------------------------------
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cat.jpg"

    img = cv.imread(imgPath)

    if img is None:
        print("Image not found!")
        return

    # ---------------------------------------
    # Resize Image
    # ---------------------------------------
    scale = 50

    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)

    img = cv.resize(img, (width, height))

    # Convert to Gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ---------------------------------------
    # Create SIFT Detector
    # ---------------------------------------
    sift = cv.SIFT_create()

    # Detect Keypoints
    keypoints = sift.detect(gray, None)

    # Draw Rich Keypoints
    output = cv.drawKeypoints(
        gray,
        keypoints,
        None,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # ---------------------------------------
    # Display Results
    # ---------------------------------------
    plt.figure(figsize=(14,6))

    plt.subplot(121)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(output, cmap='gray')
    plt.title(f"SIFT Keypoints ({len(keypoints)})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    siftDetection()