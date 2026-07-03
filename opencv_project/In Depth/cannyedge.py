import cv2 as cv

def callback(x):
    pass

def cannyEdge():

    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"

    img = cv.imread(imgPath)

    if img is None:
        print("Image not found!")
        return

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Resize image (same as tutorial)
    height, width, _ = img.shape
    scale = 1 / 5

    heightScale = int(height * scale)
    widthScale = int(width * scale)

    img = cv.resize(
        img,
        (widthScale, heightScale),
        interpolation=cv.INTER_LINEAR
    )

    winname = "Canny Edge"

    cv.namedWindow(winname)

    cv.createTrackbar("Min Threshold", winname, 50, 255, callback)
    cv.createTrackbar("Max Threshold", winname, 150, 255, callback)

    while True:

        minThres = cv.getTrackbarPos("Min Threshold", winname)
        maxThres = cv.getTrackbarPos("Max Threshold", winname)

        # Ensure max threshold is never less than min threshold
        if maxThres <= minThres:
            maxThres = minThres + 1

        canny = cv.Canny(img, minThres, maxThres)

        cv.imshow(winname, canny)

        key = cv.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:   # q or ESC
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    cannyEdge()