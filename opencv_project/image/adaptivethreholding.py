import cv2

# Read image
img = cv2.imread("image1.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------
# Adaptive Mean Threshold
# Threshold is calculated using the mean of neighboring pixels.
# --------------------------------------------------
adaptive_mean = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

# --------------------------------------------------
# Adaptive Gaussian Threshold
# Threshold is calculated using a weighted Gaussian neighborhood.
# --------------------------------------------------
adaptive_gaussian = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

cv2.imshow("Adaptive Mean", adaptive_mean)
cv2.imshow("Adaptive Gaussian", adaptive_gaussian)

cv2.waitKey(0)
cv2.destroyAllWindows()