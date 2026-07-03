import cv2

# Read the image
img = cv2.imread("image1.jpg")

if img is None:
    print("Error: image1.jpg not found!")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Original Image", img)
cv2.imshow("Grayscale Image", gray)

# --------------------------------------------------
# 1. Binary Threshold
# Pixels above the threshold become white, others become black.
# --------------------------------------------------
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary Threshold", binary)

# --------------------------------------------------
# 2. Binary Inverse Threshold
# Pixels above the threshold become black, others become white.
# --------------------------------------------------
ret, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Binary Inverse", binary_inv)

# --------------------------------------------------
# 3. Truncate Threshold
# Pixels above the threshold are set to the threshold value.
# --------------------------------------------------
ret, trunc = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow("Truncate Threshold", trunc)

# --------------------------------------------------
# 4. To Zero Threshold
# Pixels below the threshold become zero; others remain unchanged.
# --------------------------------------------------
ret, tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow("To Zero Threshold", tozero)

# --------------------------------------------------
# 5. To Zero Inverse Threshold
# Pixels above the threshold become zero; others remain unchanged.
# --------------------------------------------------
ret, tozero_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow("To Zero Inverse", tozero_inv)

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()