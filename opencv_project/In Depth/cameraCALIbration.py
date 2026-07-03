import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Image Paths
# ============================================================
imagePaths = [
    r"C:\Users\dronr\Documents\opencv_project\chessboard.jpg",
    r"C:\Users\dronr\Documents\opencv_project\chessboard2.jpg"
]

# ============================================================
# Chessboard Size (INNER CORNERS)
# ============================================================
nRows = 7
nCols = 7

# ============================================================
# Corner Refinement Criteria
# ============================================================
criteria = (
    cv.TERM_CRITERIA_EPS +
    cv.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ============================================================
# Object Points
# ============================================================
objp = np.zeros((nRows * nCols, 3), np.float32)
objp[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)

objectPoints = []
imagePoints = []

imgSize = None

# ============================================================
# Find Chessboard Corners
# ============================================================
for imgPath in imagePaths:

    img = cv.imread(imgPath)

    if img is None:
        print(f"Cannot read image:\n{imgPath}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    imgSize = gray.shape[::-1]

    flags = (
        cv.CALIB_CB_ADAPTIVE_THRESH |
        cv.CALIB_CB_NORMALIZE_IMAGE
    )

    found, corners = cv.findChessboardCorners(
        gray,
        (nRows, nCols),
        flags
    )

    if found:

        print(f"[OK] Chessboard Found : {os.path.basename(imgPath)}")

        corners2 = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        objectPoints.append(objp)
        imagePoints.append(corners2)

        cv.drawChessboardCorners(
            img,
            (nRows, nCols),
            corners2,
            found
        )

        cv.imshow("Detected Chessboard", img)
        cv.waitKey(1000)

    else:

        print(f"[FAILED] Chessboard NOT Found : {os.path.basename(imgPath)}")

cv.destroyAllWindows()

# ============================================================
# Check Images
# ============================================================
if len(objectPoints) < 2:

    print("\n====================================")
    print("Calibration Failed")
    print("Reason : Less than two valid images.")
    print("====================================")

    exit()

# ============================================================
# Camera Calibration
# ============================================================
ret, cameraMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
    objectPoints,
    imagePoints,
    imgSize,
    None,
    None
)

print("\n===================================")
print("Calibration Successful")
print("===================================")

print("\nReprojection Error")
print(ret)

print("\nCamera Matrix")
print(cameraMatrix)

print("\nDistortion Coefficients")
print(distCoeff)

# ============================================================
# Save Calibration
# ============================================================
savePath = r"C:\Users\dronr\Desktop\wao\calibration.npz"

np.savez(
    savePath,
    cameraMatrix=cameraMatrix,
    distCoeff=distCoeff,
    rvecs=rvecs,
    tvecs=tvecs
)

print("\nCalibration saved to:")
print(savePath)

# ============================================================
# Undistort First Image
# ============================================================
img = cv.imread(imagePaths[0])

h, w = img.shape[:2]

newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
    cameraMatrix,
    distCoeff,
    (w, h),
    1,
    (w, h)
)

undistorted = cv.undistort(
    img,
    cameraMatrix,
    distCoeff,
    None,
    newCameraMatrix
)

# ============================================================
# Crop ROI
# ============================================================
x, y, w, h = roi

if w > 0 and h > 0:
    undistorted = undistorted[y:y+h, x:x+w]

# ============================================================
# Display
# ============================================================
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(122)
plt.imshow(cv.cvtColor(undistorted, cv.COLOR_BGR2RGB))
plt.title("Undistorted")
plt.axis("off")

plt.tight_layout()
plt.show()