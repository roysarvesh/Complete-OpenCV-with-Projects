import cv2 as cv

def callback(x):
    pass

def imageBlending():
    img1Path = r"C:\Users\dronr\Documents\opencv_project\cat.jpg"
    img2Path = r"C:\Users\dronr\Documents\opencv_project\cats 2.jpg"

    img1 = cv.imread(img1Path)
    img2 = cv.imread(img2Path)

    if img1 is None or img2 is None:
        print("Error: Could not load one or both images.")
        return

    height, width = img1.shape[:2]

    # Resize second image to match first image
    img2 = cv.resize(img2, (width, height))

    windowName = "Image Blending"
    cv.namedWindow(windowName)

    scale = 100
    cv.createTrackbar("alpha", windowName, 0, scale, callback)
    cv.createTrackbar("gamma", windowName, 0, 255, callback)

    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        alpha = cv.getTrackbarPos("alpha", windowName) / scale
        beta = 1 - alpha
        gamma = cv.getTrackbarPos("gamma", windowName)

        imgBlend = cv.addWeighted(img1, alpha, img2, beta, gamma)

        cv.imshow(windowName, imgBlend)

    cv.destroyAllWindows()

if __name__ == "__main__":
    imageBlending()