import cv2

# Read image
img = cv2.imread("image.jpg")
if img is None:
    print("Image not found!")
    exit()
# -------------------------
# BGR to Grayscale
# -------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------
# BGR to HSV
# -------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# -------------------------
# BGR to LAB
# -------------------------
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# -------------------------
# BGR to RGB
# -------------------------
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------
# HSV to BGR
# -------------------------
hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# -------------------------
# LAB to BGR
# -------------------------
lab_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# -------------------------
# Display Images
# -------------------------
cv2.imshow("Original (BGR)", img)
cv2.imshow("Gray", gray)
cv2.imshow("HSV", hsv)
cv2.imshow("LAB", lab)
cv2.imshow("RGB", rgb)
cv2.imshow("HSV -> BGR", hsv_bgr)
cv2.imshow("LAB -> BGR", lab_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()