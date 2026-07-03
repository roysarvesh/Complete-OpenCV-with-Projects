import cv2
import numpy as np

# Read the image
img = cv2.imread("image1.jpg")

if img is None:
    print("Error: image1.jpg not found!")
    exit()

# Display Original Image
cv2.imshow("Original Image", img)

# --------------------------------------------------
# Create a blank mask (black image)
# Black (0) hides the image and White (255) reveals it.
# --------------------------------------------------
blank = np.zeros(img.shape[:2], dtype='uint8')

# --------------------------------------------------
# Draw a white rectangle on the mask
# Only this rectangular region will be visible.
# --------------------------------------------------
mask = cv2.rectangle(blank, (100, 100), (500, 400), 255, -1)

# Display the mask
cv2.imshow("Rectangle Mask", mask)

# --------------------------------------------------
# Apply the mask to the original image
# Only the white region of the mask is kept.
# --------------------------------------------------
masked = cv2.bitwise_and(img, img, mask=mask)

# Display the masked image
cv2.imshow("Masked Image", masked)

cv2.waitKey(0)
cv2.destroyAllWindows()