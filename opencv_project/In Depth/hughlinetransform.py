import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Read Image
# -----------------------------
imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"

img = cv.imread(imgPath)

if img is None:
    print("Error: Could not load image.")
    exit()

# Convert BGR to RGB for matplotlib
imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# -----------------------------
# Preprocessing
# -----------------------------
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Gaussian Blur
blur = cv.GaussianBlur(gray, (5, 5), 1)

# Canny Edge Detection
edges = cv.Canny(blur, 100, 200)

# -----------------------------
# Hough Line Transform
# -----------------------------
distResolution = 1                  # Distance resolution in pixels
angleResolution = np.pi / 180       # Angle resolution (1 degree)
threshold = 150                     # Minimum votes

lines = cv.HoughLines(
    edges,
    distResolution,
    angleResolution,
    threshold
)

# Copy image for drawing
result = imgRGB.copy()

# -----------------------------
# Draw Lines
# -----------------------------
if lines is not None:

    k = 3000

    for currentLine in lines:

        rho, theta = currentLine[0]

        # Unit normal vector
        dHat = np.array([[np.cos(theta)],
                         [np.sin(theta)]])

        # Closest point on the line
        d = rho * dHat

        # Direction vector of the line
        lHat = np.array([[-np.sin(theta)],
                         [ np.cos(theta)]])

        # Two far-away points
        p1 = d + k * lHat
        p2 = d - k * lHat

        p1 = p1.astype(int)
        p2 = p2.astype(int)

        cv.line(
            result,
            (p1[0][0], p1[1][0]),
            (p2[0][0], p2[1][0]),
            (255, 0, 0),      # Blue in RGB
            2
        )

# -----------------------------
# Display Results
# -----------------------------
plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
plt.imshow(imgRGB)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(blur, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edges")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(result)
plt.title("Hough Line Transform")
plt.axis("off")

plt.tight_layout()
plt.show()