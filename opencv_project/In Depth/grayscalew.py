import cv2 as cv


def grayscale():
    img = cv.imread(r'C:\Users\dronr\Documents\opencv_project\cats 2.jpg')

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow('Gray', imgGray)
    cv.waitKey(0)
    cv.destroyAllWindows()


def readAsGray():
    img = cv.imread(
        r'C:\Users\dronr\Documents\opencv_project\cats 2.jpg',
        cv.IMREAD_GRAYSCALE
    )

    cv.imshow('Gray', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    grayscale()
    readAsGray()