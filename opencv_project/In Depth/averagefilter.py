import cv2 as cv

def callback(x):
    pass

def averageFiltering():

    # Read image
    imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"
    img = cv.imread(imgPath)

    if img is None:
        print("Error: Could not load image.")
        return

    winName = "Average Filter"

    cv.namedWindow(winName)
    cv.createTrackbar("Kernel Size", winName, 1, 100, callback)

    # Resize image for display
    height, width = img.shape[:2]
    scale = 0.5

    width = int(width * scale)
    height = int(height * scale)

    img = cv.resize(img, (width, height))

    while True:

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        n = cv.getTrackbarPos("Kernel Size", winName)

        # Kernel size must be at least 1 and odd
        if n < 1:
            n = 1
        if n % 2 == 0:
            n += 1

        # Apply Average Filter
        imgFilter = cv.blur(img, (n, n))

        cv.imshow(winName, imgFilter)

    cv.destroyAllWindows()


if __name__ == "__main__":
    averageFiltering()