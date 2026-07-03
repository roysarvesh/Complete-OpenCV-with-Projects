import cv2 as cv
import numpy as np

# ======================================================
# Video Path
# ======================================================
videoPath = r"C:\Users\dronr\Documents\opencv_project\Videos\dog.mp4"

# ======================================================
# Lucas-Kanade Optical Flow (Sparse)
# ======================================================
def lucasKanade():

    cap = cv.VideoCapture(videoPath)

    ret, frame = cap.read()

    if not ret:
        print("Cannot open video.")
        return

    prevGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Shi-Tomasi Parameters
    featureParams = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # Lucas-Kanade Parameters
    lkParams = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(
            cv.TERM_CRITERIA_EPS |
            cv.TERM_CRITERIA_COUNT,
            10,
            0.03
        )
    )

    prevPts = cv.goodFeaturesToTrack(
        prevGray,
        mask=None,
        **featureParams
    )

    mask = np.zeros_like(frame)

    colors = np.random.randint(0,255,(100,3))

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        nextPts, status, err = cv.calcOpticalFlowPyrLK(
            prevGray,
            gray,
            prevPts,
            None,
            **lkParams
        )

        if nextPts is not None:

            goodNew = nextPts[status==1]
            goodOld = prevPts[status==1]

            for i,(new,old) in enumerate(zip(goodNew,goodOld)):

                xNew,yNew = new.ravel()
                xOld,yOld = old.ravel()

                mask = cv.line(
                    mask,
                    (int(xOld),int(yOld)),
                    (int(xNew),int(yNew)),
                    colors[i].tolist(),
                    2
                )

                frame = cv.circle(
                    frame,
                    (int(xNew),int(yNew)),
                    5,
                    colors[i].tolist(),
                    -1
                )

            output = cv.add(frame,mask)

            cv.imshow("Lucas-Kanade Optical Flow",output)

            prevGray = gray.copy()
            prevPts = goodNew.reshape(-1,1,2)

        key = cv.waitKey(30)

        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()


# ======================================================
# Farneback Optical Flow (Dense)
# ======================================================
def farneback():

    cap = cv.VideoCapture(videoPath)

    ret, frame = cap.read()

    if not ret:
        print("Cannot open video.")
        return

    prevGray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame)
    hsv[:,:,1] = 255

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(
            prevGray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN
        )

        mag, ang = cv.cartToPolar(
            flow[:,:,0],
            flow[:,:,1]
        )

        hsv[:,:,0] = ang * 180 / np.pi / 2

        hsv[:,:,2] = cv.normalize(
            mag,
            None,
            0,
            255,
            cv.NORM_MINMAX
        )

        output = cv.cvtColor(
            hsv,
            cv.COLOR_HSV2BGR
        )

        cv.imshow("Farneback Optical Flow",output)

        prevGray = gray

        key = cv.waitKey(30)

        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":

    # Uncomment ONE at a time

    lucasKanade()

    # farneback()