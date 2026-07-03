import cv2 as cv


def drawCircle(event, x, y, flags, param):
    img = param

    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img, (x, y), 10, (0, 0, 255), -1)


def doubleClickDrawing():
    img = cv.imread(r'C:\Users\dronr\Documents\opencv_project\cats.jpg')

    windowName = 'Drawing App'

    cv.namedWindow(windowName)
    cv.setMouseCallback(windowName, drawCircle, img)

    while True:
        cv.imshow(windowName, img)

        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()


class DrawingApp:

    def __init__(self, imagePath):
        self.imagePath = imagePath
        self.startX = 0
        self.startY = 0
        self.isDrawing = False

    def drawLine(self, event, x, y, flags, param):
        img = param

        if event == cv.EVENT_LBUTTONDOWN:
            self.isDrawing = True
            self.startX, self.startY = x, y

        elif event == cv.EVENT_MOUSEMOVE and self.isDrawing:
            cv.line(img,
                    (self.startX, self.startY),
                    (x, y),
                    (255, 255, 255),
                    3)

        elif event == cv.EVENT_LBUTTONUP:
            self.isDrawing = False
            cv.line(img,
                    (self.startX, self.startY),
                    (x, y),
                    (255, 255, 255),
                    3)

    def run(self):
        img = cv.imread(self.imagePath)

        windowName = 'Drawing App'

        cv.namedWindow(windowName)
        cv.setMouseCallback(windowName, self.drawLine, img)

        while True:
            cv.imshow(windowName, img)

            if cv.waitKey(1) == ord('q'):
                break

        cv.destroyAllWindows()


def holdAndDragDrawing():
    imgPath = r'C:\Users\dronr\Documents\opencv_project\cats.jpg'

    app = DrawingApp(imgPath)
    app.run()


if __name__ == '__main__':
    # doubleClickDrawing()
    holdAndDragDrawing()