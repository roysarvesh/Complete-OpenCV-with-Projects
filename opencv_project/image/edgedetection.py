#gradient
import cv2
import numpy as np
img = cv2.imread("image1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original Image", img)
cv2.imshow("Grayscale Image", gray)
# --------------------------------------------------
# 1. Laplacian Gradient
# Detects edges in all directions using the second derivative.
# --------------------------------------------------
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

cv2.imshow("Laplacian Gradient", laplacian)

# --------------------------------------------------
# 2. Sobel X Gradient
# Detects vertical edges (changes along the X-direction).
# --------------------------------------------------
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobelx = np.uint8(np.absolute(sobelx))

cv2.imshow("Sobel X", sobelx)

# --------------------------------------------------
# 3. Sobel Y Gradient
# Detects horizontal edges (changes along the Y-direction).
# --------------------------------------------------
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
sobely = np.uint8(np.absolute(sobely))

cv2.imshow("Sobel Y", sobely)

# --------------------------------------------------
# 4. Combined Sobel Gradient
# Combines Sobel X and Sobel Y to detect edges in both directions.
# --------------------------------------------------
combined = cv2.bitwise_or(sobelx, sobely)

cv2.imshow("Combined Sobel", combined)

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()