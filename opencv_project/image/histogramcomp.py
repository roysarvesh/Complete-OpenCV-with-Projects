#Gray Histogram
"""import cv2
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread("image1.jpg")

if img is None:
    print("Error: image1.jpg not found!")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Compute histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Display image
cv2.imshow("Grayscale Image", gray)

# Plot histogram
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.plot(hist, color='black')
plt.xlim([0, 256])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()"""
#Color Histogram (BGR)
"""import cv2
from matplotlib import pyplot as plt

# Read image
img = cv2.imread("image1.jpg")

if img is None:
    print("Error: image1.jpg not found!")
    exit()

cv2.imshow("Original Image", img)

# Colors for plotting
colors = ('b', 'g', 'r')

plt.figure()
plt.title("Color Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")

# Compute histogram for each channel
for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()"""""
#histogram masking 
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image
img = cv2.imread("image1.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a blank mask
blank = np.zeros(gray.shape, dtype='uint8')

# Draw a white rectangle mask
mask = cv2.rectangle(blank, (100, 100), (500, 400), 255, -1)

# Apply mask
masked = cv2.bitwise_and(gray, gray, mask=mask)

# Histogram of masked region
hist = cv2.calcHist([gray], [0], mask, [256], [0, 256])

cv2.imshow("Mask", mask)
cv2.imshow("Masked Image", masked)

plt.title("Histogram of Masked Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.plot(hist, color='black')
plt.xlim([0, 256])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()