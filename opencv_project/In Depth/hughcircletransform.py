import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------
# Read Image
# ----------------------------------
imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"

img = cv.imread(imgPath)

if img is None:
    print("Image not found!")
    exit()

imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# ----------------------------------
# Median Blur (recommended for circles)
# ----------------------------------
grayBlur = cv.medianBlur(gray, 5)

# ----------------------------------
# Hough Circle Transform
# ----------------------------------
circles = cv.HoughCircles(
    grayBlur,
    cv.HOUGH_GRADIENT,
    dp=1.2,
    minDist=40,
    param1=100,
    param2=25,
    minRadius=60,
    maxRadius=200
)

output = imgRGB.copy()

# ----------------------------------
# Draw Circles
# ----------------------------------
if circles is not None:

    circles = np.uint16(np.around(circles))

    for circle in circles[0, :]:

        x = circle[0]
        y = circle[1]
        r = circle[2]

        # Draw outer circle
        cv.circle(output, (x, y), r, (0, 255, 0), 3)

        # Draw center
        cv.circle(output, (x, y), 3, (255, 0, 0), -1)

# ----------------------------------
# Display
# ----------------------------------
plt.figure(figsize=(15,5))

plt.subplot(131)
plt.imshow(imgRGB)
plt.title("Original")
plt.axis("off")

plt.subplot(132)
plt.imshow(grayBlur, cmap="gray")
plt.title("Median Blur")
plt.axis("off")

plt.subplot(133)
plt.imshow(output)
plt.title("Detected Circles")
plt.axis("off")

plt.tight_layout()
plt.show()