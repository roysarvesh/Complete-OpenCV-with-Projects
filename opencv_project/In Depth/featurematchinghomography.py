import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Image Paths
imgPath1 = r"C:\Users\dronr\Documents\opencv_project\teslalogo.png"
imgPath2 = r"C:\Users\dronr\Documents\opencv_project\teslacar.jpg"

# Read Images
img1 = cv.imread(imgPath1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(imgPath2, cv.IMREAD_GRAYSCALE)

# Color copy for drawing
img2_color = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

# SIFT
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN Parameters
FLANN_INDEX_KDTREE = 1

index_params = dict(
    algorithm=FLANN_INDEX_KDTREE,
    trees=5
)

search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Lowe Ratio Test
good = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1,1,2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1,1,2)

    # Homography
    M, mask = cv.findHomography(
        src_pts,
        dst_pts,
        cv.RANSAC,
        5.0
    )

    matchesMask = mask.ravel().tolist()

    h, w = img1.shape

    pts = np.float32([
        [0,0],
        [0,h-1],
        [w-1,h-1],
        [w-1,0]
    ]).reshape(-1,1,2)

    dst = cv.perspectiveTransform(pts, M)

    img2_color = cv.polylines(
        img2_color,
        [np.int32(dst)],
        True,
        (0,255,0),
        3
    )

else:

    print("Not enough matches.")

    matchesMask = None

draw_params = dict(
    matchColor=(0,255,0),
    singlePointColor=None,
    matchesMask=matchesMask,
    flags=2
)

result = cv.drawMatches(
    img1,
    kp1,
    img2_color,
    kp2,
    good,
    None,
    **draw_params
)

plt.figure(figsize=(15,8))
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title("Feature Matching with Homography")
plt.axis("off")
plt.show()