import cv2 as cv
import matplotlib.pyplot as plt

# Read image
imgPath = r"C:\Users\dronr\Documents\opencv_project\park.jpg"
img = cv.imread(imgPath)

if img is None:
    print("Image not found!")
    exit()

# Convert BGR to RGB for Matplotlib
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.figure(figsize=(14, 8))

# Original Image
plt.subplot(231)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

# -----------------------------
# Select ROI (Change these values according to your image)
# Format: img[y1:y2, x1:x2]
# -----------------------------
imgRegion = img[100:300, 100:300]

plt.subplot(232)
plt.imshow(imgRegion)
plt.title("ROI")
plt.axis("off")

# Convert ROI to HSV
imgRegionHSV = cv.cvtColor(imgRegion, cv.COLOR_RGB2HSV)

# Calculate 2D Histogram (H and S channels)
imgRegionHist = cv.calcHist(
    [imgRegionHSV],
    [0, 1],
    None,
    [180, 256],
    [0, 180, 0, 256]
)

# Normalize histogram
cv.normalize(imgRegionHist, imgRegionHist, 0, 255, cv.NORM_MINMAX)

# Convert full image to HSV
imgHSV = cv.cvtColor(img, cv.COLOR_RGB2HSV)

# Histogram Backprojection
out = cv.calcBackProject(
    [imgHSV],
    [0, 1],
    imgRegionHist,
    [0, 180, 0, 256],
    1
)

plt.subplot(233)
plt.imshow(out, cmap="gray")
plt.title("Back Projection")
plt.axis("off")

# Smooth result
ellipseKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
cv.filter2D(out, -1, ellipseKernel, out)

plt.subplot(234)
plt.imshow(out, cmap="gray")
plt.title("Smoothed")
plt.axis("off")

# Threshold
_, mask = cv.threshold(out, 0, 255, cv.THRESH_BINARY)

plt.subplot(235)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.axis("off")

# Segment Image
maskAllChannels = cv.merge((mask, mask, mask))
imgSeg = cv.bitwise_and(img, maskAllChannels)

plt.subplot(236)
plt.imshow(imgSeg)
plt.title("Segmented Image")
plt.axis("off")

plt.tight_layout()
plt.show()