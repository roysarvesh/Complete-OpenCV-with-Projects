import cv2 as cv
import numpy as np

# -----------------------------
# Change to "meanshift" or "camshift"
# -----------------------------
option = "camshift"

videoPath = r"C:\Users\dronr\Documents\opencv_project\Videos\kitten.mp4"

cap = cv.VideoCapture(videoPath)

ret, frame = cap.read()

if not ret:
    print("Cannot read video.")
    exit()

# Select ROI
roi = cv.selectROI("Select Object", frame, False, False)
cv.destroyWindow("Select Object")

x, y, w, h = roi

# ROI for Histogram
roiImg = frame[y:y+h, x:x+w]

# Convert ROI to HSV
hsvROI = cv.cvtColor(roiImg, cv.COLOR_BGR2HSV)

# Ignore low light pixels
mask = cv.inRange(
    hsvROI,
    np.array((0., 60., 32.)),
    np.array((180., 255., 255.))
)

# Histogram
roiHist = cv.calcHist([hsvROI], [0], mask, [180], [0, 180])

cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)

# Termination Criteria
termCrit = (
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
    10,
    1
)

trackWindow = (x, y, w, h)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    backProj = cv.calcBackProject(
        [hsv],
        [0],
        roiHist,
        [0, 180],
        1
    )

    # --------------------------
    # MeanShift
    # --------------------------
    if option == "meanshift":

        ret, trackWindow = cv.meanShift(
            backProj,
            trackWindow,
            termCrit
        )

        x, y, w, h = trackWindow

        cv.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # --------------------------
    # CamShift
    # --------------------------
    elif option == "camshift":

        ret, trackWindow = cv.CamShift(
            backProj,
            trackWindow,
            termCrit
        )

        pts = cv.boxPoints(ret)
        pts = np.int32(pts)

        cv.polylines(
            frame,
            [pts],
            True,
            (0, 255, 0),
            2
        )

    cv.imshow("Tracking", frame)

    key = cv.waitKey(30) & 0xFF

    if key == 27:
        break

cap.release()
cv.destroyAllWindows()