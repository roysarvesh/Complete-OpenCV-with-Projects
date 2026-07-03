import cv2 as cv
import matplotlib.pyplot as plt

def FLANN_FeatureMatching():

    # Image Paths
    imgPath1 = r"C:\Users\dronr\Documents\opencv_project\teslalogo.png"
    imgPath2 = r"C:\Users\dronr\Documents\opencv_project\teslalogo2.jpg"

    # Read Images
    img1 = cv.imread(imgPath1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(imgPath2, cv.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error loading images.")
        return

    # Create SIFT Detector
    sift = cv.SIFT_create()

    # Detect Keypoints and Compute Descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN Parameters
    FLANN_INDEX_KDTREE = 1

    index_params = dict(
        algorithm=FLANN_INDEX_KDTREE,
        trees=5
    )

    search_params = dict(
        checks=50
    )

    # Create FLANN Matcher
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # KNN Matching
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's Ratio Test
    good_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print("Keypoints Image 1 :", len(kp1))
    print("Keypoints Image 2 :", len(kp2))
    print("Good Matches      :", len(good_matches))

    # Draw Matches
    result = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(15,7))
    plt.imshow(result, cmap='gray')
    plt.title("FLANN Feature Matching using SIFT")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    FLANN_FeatureMatching()