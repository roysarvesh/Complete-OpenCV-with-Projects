import cv2 as cv
import matplotlib.pyplot as plt

def ORB():

    # Read Image
    imgPath = r"C:\Users\dronr\Documents\opencv_project\teslalogo.png"

    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    if imgGray is None:
        print("Image not found!")
        return

    # Resize Image (Optional)
    imgGray = cv.resize(imgGray, (600, 600))

    # Create ORB Object
    orb = cv.ORB_create()

    # Detect Keypoints
    keypoints = orb.detect(imgGray, None)

    # Compute Descriptors
    keypoints, descriptors = orb.compute(imgGray, keypoints)

    # Draw Rich Keypoints
    imgORB = cv.drawKeypoints(
        imgGray,
        keypoints,
        None,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Print Information
    print("Number of Keypoints :", len(keypoints))

    if descriptors is not None:
        print("Descriptor Shape :", descriptors.shape)
        print("Descriptor Size  :", descriptors.shape[1], "bytes")
        print("\nFirst Descriptor:\n")
        print(descriptors[0])
    else:
        print("No descriptors found.")

    # Display Image
    plt.figure(figsize=(8,8))
    plt.imshow(imgORB, cmap='gray')
    plt.title("ORB Feature Detection")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    ORB()