import cv2 as cv
import matplotlib.pyplot as plt


def FASTCornerDetection():

    # ---------------------------------------
    # Read Image
    # ---------------------------------------
    imgPath = r"C:\Users\dronr\Documents\opencv_project\coin.jpg"

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

    # Convert BGR to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Convert to Gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ---------------------------------------
    # Create FAST Detector
    # ---------------------------------------
    fast = cv.FastFeatureDetector_create()

    # Parameters
    threshold = 40
    fast.setThreshold(threshold)

    fast.setNonmaxSuppression(True)

    # Detect Keypoints
    keypoints = fast.detect(gray, None)

    # Draw Keypoints
    output = cv.drawKeypoints(
        imgRGB,
        keypoints,
        None,
        color=(255, 0, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # ---------------------------------------
    # Print Information
    # ---------------------------------------
    print("Threshold :", fast.getThreshold())
    print("Nonmax Suppression :", fast.getNonmaxSuppression())
    print("Number of Keypoints :", len(keypoints))

    # ---------------------------------------
    # Display Results
    # ---------------------------------------
    plt.figure(figsize=(14,6))

    plt.subplot(121)
    plt.imshow(imgRGB)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(output)
    plt.title(f"FAST Keypoints ({len(keypoints)})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    FASTCornerDetection()