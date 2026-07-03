import cv2 as cv


def trackbarCallback(x):
    pass


def trackbar():
    img = cv.imread(r'C:\Users\dronr\Documents\opencv_project\cats.jpg')

    windowName = 'Trackbar Demo'

    cv.namedWindow(windowName)

    cv.createTrackbar('B', windowName, 0, 255, trackbarCallback)
    cv.createTrackbar('G', windowName, 0, 255, trackbarCallback)
    cv.createTrackbar('R', windowName, 0, 255, trackbarCallback)

    while True:
        temp = img.copy()

        b = cv.getTrackbarPos('B', windowName)
        g = cv.getTrackbarPos('G', windowName)
        r = cv.getTrackbarPos('R', windowName)

        cv.circle(temp, (496, 325), 10, (b, g, r), -1)
        cv.circle(temp, (353, 315), 10, (b, g, r), -1)

        cv.imshow(windowName, temp)

        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    trackbar()