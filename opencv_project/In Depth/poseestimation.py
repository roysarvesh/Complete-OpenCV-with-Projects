import cv2 as cv
import numpy as np

# ==========================================================
# Load Calibration
# ==========================================================

calibrationPath = r"C:\Users\dronr\Desktop\wao\calibration.npz"

data = np.load(calibrationPath)

cameraMatrix = data["cameraMatrix"]
distCoeff = data["distCoeff"]

# ==========================================================
# Load Chessboard Image
# ==========================================================

imgPath = r"C:\Users\dronr\Documents\opencv_project\chessboard.jpg"

img = cv.imread(imgPath)

if img is None:
    print("Image not found!")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# ==========================================================
# Chessboard Size (Inner Corners)
# ==========================================================

rows = 7
cols = 7

# ==========================================================
# Corner Refinement Criteria
# ==========================================================

criteria = (
    cv.TERM_CRITERIA_EPS +
    cv.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ==========================================================
# Prepare Object Points
# ==========================================================

objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# ==========================================================
# Cube Coordinates
# ==========================================================

cube = np.float32([
    [0,0,0],
    [0,3,0],
    [3,3,0],
    [3,0,0],

    [0,0,-3],
    [0,3,-3],
    [3,3,-3],
    [3,0,-3]
])

# ==========================================================
# Find Chessboard Corners
# ==========================================================

flags = (
    cv.CALIB_CB_ADAPTIVE_THRESH |
    cv.CALIB_CB_NORMALIZE_IMAGE
)

found, corners = cv.findChessboardCorners(
    gray,
    (rows, cols),
    flags
)

if not found:
    print("Chessboard not detected.")
    exit()

print("Chessboard Detected!")

# ==========================================================
# Refine Corners
# ==========================================================

corners2 = cv.cornerSubPix(
    gray,
    corners,
    (11,11),
    (-1,-1),
    criteria
)

# ==========================================================
# Solve PnP
# ==========================================================

success, rvec, tvec = cv.solvePnP(
    objectPoints,
    corners2,
    cameraMatrix,
    distCoeff
)

print("\nPose Estimation Successful :", success)

print("\nRotation Vector")
print(rvec)

print("\nTranslation Vector")
print(tvec)

# ==========================================================
# Project Cube
# ==========================================================

imgPoints, _ = cv.projectPoints(
    cube,
    rvec,
    tvec,
    cameraMatrix,
    distCoeff
)

imgPoints = np.int32(imgPoints).reshape(-1,2)

# ==========================================================
# Draw Cube
# ==========================================================

def drawCube(image, pts):

    # Bottom Square
    image = cv.drawContours(
        image,
        [pts[:4]],
        -1,
        (0,255,0),
        3
    )

    # Vertical Edges
    for i, j in zip(range(4), range(4,8)):
        image = cv.line(
            image,
            tuple(pts[i]),
            tuple(pts[j]),
            (255,0,0),
            3
        )

    # Top Square
    image = cv.drawContours(
        image,
        [pts[4:]],
        -1,
        (0,0,255),
        3
    )

    return image

img = drawCube(img, imgPoints)

# ==========================================================
# Draw Chessboard Corners
# ==========================================================

cv.drawChessboardCorners(
    img,
    (rows, cols),
    corners2,
    found
)

# ==========================================================
# Display
# ==========================================================

cv.namedWindow("Pose Estimation", cv.WINDOW_NORMAL)
cv.imshow("Pose Estimation", img)

cv.waitKey(0)
cv.destroyAllWindows()