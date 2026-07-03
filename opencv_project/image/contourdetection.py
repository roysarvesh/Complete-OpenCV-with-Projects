import cv2
import numpy as np

# Read the image
img = cv2.imread("image.jpg")

if img is None:
    print("Error: image.jpg not found!")
    exit()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blur, 125, 175)
contours, hierarchy = cv2.findContours(
    canny,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"{len(contours)} contour(s) found.")
contour_img = img.copy()
cv2.drawContours(
    contour_img,
    contours,
    -1,                 # Draw all contours
    (0, 255, 0),        # Green color
    2                   # Thickness
)

cv2.imshow("Original Image", img)
cv2.imshow("Gray", gray)
cv2.imshow("Canny Edge", canny)
cv2.imshow("Contours", contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()