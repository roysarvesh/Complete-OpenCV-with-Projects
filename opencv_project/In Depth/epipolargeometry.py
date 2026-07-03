import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Image Paths
# ==========================================================

leftPath = r"C:\Users\dronr\Documents\opencv_project\left.png"
rightPath = r"C:\Users\dronr\Documents\opencv_project\right.jpg"

# ==========================================================
# Read Images
# ==========================================================

imgLeft = cv.imread(leftPath, cv.IMREAD_GRAYSCALE)
imgRight = cv.imread(rightPath, cv.IMREAD_GRAYSCALE)

if imgLeft is None or imgRight is None:
    print("Images not found!")
    exit()

# ==========================================================
# SIFT Feature Detection
# ==========================================================

sift = cv.SIFT_create()

kpLeft, desLeft = sift.detectAndCompute(imgLeft, None)
kpRight, desRight = sift.detectAndCompute(imgRight, None)

print("Left Keypoints :", len(kpLeft))
print("Right Keypoints:", len(kpRight))

# ==========================================================
# FLANN Matcher
# ==========================================================

FLANN_INDEX_KDTREE = 1

index_params = dict(
    algorithm=FLANN_INDEX_KDTREE,
    trees=5
)

search_params = dict(
    checks=50
)

flann = cv.FlannBasedMatcher(
    index_params,
    search_params
)

matches = flann.knnMatch(
    desLeft,
    desRight,
    k=2
)

# ==========================================================
# Lowe Ratio Test
# ==========================================================

ptsLeft = []
ptsRight = []

for m, n in matches:

    if m.distance < 0.75 * n.distance:

        ptsLeft.append(kpLeft[m.queryIdx].pt)
        ptsRight.append(kpRight[m.trainIdx].pt)

ptsLeft = np.int32(ptsLeft)
ptsRight = np.int32(ptsRight)

print("Good Matches :", len(ptsLeft))

# ==========================================================
# Fundamental Matrix
# ==========================================================

F, mask = cv.findFundamentalMat(
    ptsLeft,
    ptsRight,
    cv.FM_RANSAC
)

ptsLeft = ptsLeft[mask.ravel() == 1]
ptsRight = ptsRight[mask.ravel() == 1]

print("\nFundamental Matrix\n")
print(F)

# ==========================================================
# Draw Epilines
# ==========================================================

def drawLines(img1, img2, lines, pts1, pts2):

    r, c = img1.shape

    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

    np.random.seed(10)

    for line, pt1, pt2 in zip(lines, pts1, pts2):

        color = tuple(np.random.randint(0,255,3).tolist())

        x0, y0 = map(int, [0, -line[2]/line[1]])
        x1, y1 = map(int, [c, -(line[2]+line[0]*c)/line[1]])

        img1 = cv.line(
            img1,
            (x0,y0),
            (x1,y1),
            color,
            1
        )

        img1 = cv.circle(
            img1,
            tuple(pt1),
            5,
            color,
            -1
        )

        img2 = cv.circle(
            img2,
            tuple(pt2),
            5,
            color,
            -1
        )

    return img1, img2

# ==========================================================
# Compute Epilines
# ==========================================================

linesLeft = cv.computeCorrespondEpilines(
    ptsRight.reshape(-1,1,2),
    2,
    F
)

linesLeft = linesLeft.reshape(-1,3)

img5, img6 = drawLines(
    imgLeft,
    imgRight,
    linesLeft,
    ptsLeft,
    ptsRight
)

linesRight = cv.computeCorrespondEpilines(
    ptsLeft.reshape(-1,1,2),
    1,
    F
)

linesRight = linesRight.reshape(-1,3)

img3, img4 = drawLines(
    imgRight,
    imgLeft,
    linesRight,
    ptsRight,
    ptsLeft
)

# ==========================================================
# Display
# ==========================================================

plt.figure(figsize=(14,7))

plt.subplot(121)
plt.imshow(cv.cvtColor(img5, cv.COLOR_BGR2RGB))
plt.title("Left Image with Epilines")
plt.axis("off")

plt.subplot(122)
plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
plt.title("Right Image with Epilines")
plt.axis("off")

plt.tight_layout()
plt.show()