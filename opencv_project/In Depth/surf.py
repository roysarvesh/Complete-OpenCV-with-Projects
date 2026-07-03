"""
import cv2 as cv
import matplotlib.pyplot as plt


def surfDetection():

    # ---------------------------------------
    # Read Image
    # ---------------------------------------
    imgPath = r"C:\Users\dronr\Documents\opencv_project\coin.jpg"

    gray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    if gray is None:
        print("Image not found!")
        return

    # ---------------------------------------
    # Resize Image
    # ---------------------------------------
    scale = 50

    width = int(gray.shape[1] * scale / 100)
    height = int(gray.shape[0] * scale / 100)

    gray = cv.resize(gray, (width, height))

    # ---------------------------------------
    # Create SURF Detector
    # ---------------------------------------
    hessianThreshold = 400

    surf = cv.xfeatures2d.SURF_create(
        hessianThreshold=hessianThreshold
    )

    # Detect Keypoints
    keypoints = surf.detect(gray, None)

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
    plt.title(f"SURF Keypoints ({len(keypoints)})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    surfDetection()"""