import cv2
import numpy as np
img = cv2.imread("image.jpg")
if img is None:
    print("Error: image.jpg not found!")
    exit()

# -----------------------------
# 1. Resize Image
# -----------------------------
resized = cv2.resize(img, (500, 500))

# -----------------------------
# 2. Convert BGR to Grayscale
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 3. Translation
# -----------------------------
tx, ty = 100, 50  # Shift right by 100 and down by 50

translation_matrix = np.float32([[1, 0, tx],
                                 [0, 1, ty]])

translated = cv2.warpAffine(img, translation_matrix,
                            (img.shape[1], img.shape[0]))

# -----------------------------
# 4. Rotation
# -----------------------------
(h, w) = img.shape[:2]
center = (w // 2, h // 2)

rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, rotation_matrix, (w, h))

# -----------------------------
# 5. Flip Image
# -----------------------------
horizontal_flip = cv2.flip(img, 1)
vertical_flip = cv2.flip(img, 0)
both_flip = cv2.flip(img, -1)

# -----------------------------
# 6. Crop Image
# -----------------------------
cropped = img[50:300, 100:400]

# -----------------------------
# 7. Gaussian Blur
# -----------------------------
blur = cv2.GaussianBlur(img, (7, 7), 0)

# -----------------------------
# 8. Edge Detection (Canny)
# -----------------------------
edges = cv2.Canny(img, 100, 200)

# -----------------------------
# 9. Dilating the Image
# -----------------------------
kernel = np.ones((5, 5), np.uint8)

dilated = cv2.dilate(edges, kernel, iterations=1)

# -----------------------------
# 10. Eroding the Image
# -----------------------------
eroded = cv2.erode(dilated, kernel, iterations=1)

# -----------------------------
# Display Results
# -----------------------------
cv2.imshow("Original", img)
cv2.imshow("Resized", resized)
cv2.imshow("Gray", gray)
cv2.imshow("Translated", translated)
cv2.imshow("Rotated", rotated)
cv2.imshow("Horizontal Flip", horizontal_flip)
cv2.imshow("Vertical Flip", vertical_flip)
cv2.imshow("Both Flip", both_flip)
cv2.imshow("Cropped", cropped)
cv2.imshow("Gaussian Blur", blur)
cv2.imshow("Canny Edge", edges)
cv2.imshow("Dilated", dilated)
cv2.imshow("Eroded", eroded)

cv2.waitKey(0)
cv2.destroyAllWindows()