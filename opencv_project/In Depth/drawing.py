import cv2 as cv
import numpy as np


def drawingFunctions():
    img = cv.imread(r'C:\Users\dronr\Documents\opencv_project\cats.jpg')

    white = (255, 255, 255)

    # Draw a line
    cv.line(img, (366, 402), (197, 507), white, 2)

    # Draw a rectangle
    r, c, d = img.shape
    offset = 10
    cv.rectangle(img, (offset, offset), (c - offset, r - offset), white, 8)

    # Draw a circle
    cv.circle(img, (496, 325), 10, white, -1)

    # Draw an ellipse
    cv.ellipse(img, (415, 439), (30, 20), 0, 0, 180, white, -1)

    # Draw a polygon
    pts = np.array([[234, 211],
                    [214, 71],
                    [322, 125]])

    cv.polylines(img, [pts], True, white, 3)

    # Put text
    cv.putText(img, 'TIGER', (650, 278),
               cv.FONT_HERSHEY_SIMPLEX,
               4, white, 4, cv.LINE_AA)

    cv.imshow('Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    drawingFunctions()