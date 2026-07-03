import cv2
img = cv2.imread("image1.jpg")
cv2.imshow("Original Image", img)
# --------------------------------------------------
# 1. Averaging Blur (Box Filter)
# Replaces each pixel with the average value of its neighboring pixels.
# --------------------------------------------------
average = cv2.blur(img, (7, 7))
cv2.imshow("Average Blur", average)
# --------------------------------------------------
# 2. Gaussian Blur
# Uses a Gaussian kernel to reduce noise while preserving image quality.
# --------------------------------------------------
gaussian = cv2.GaussianBlur(img, (7, 7), 0)
cv2.imshow("Gaussian Blur", gaussian)

# --------------------------------------------------
# 3. Median Blur
# Replaces each pixel with the median value of neighboring pixels; best for salt-and-pepper noise.
# --------------------------------------------------
median = cv2.medianBlur(img, 7)
cv2.imshow("Median Blur", median)

# --------------------------------------------------
# 4. Bilateral Filter
# Smooths the image while preserving edges by considering both color and distance.
# --------------------------------------------------
bilateral = cv2.bilateralFilter(img, 10, 35, 25)
cv2.imshow("Bilateral Filter", bilateral)

# Wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()