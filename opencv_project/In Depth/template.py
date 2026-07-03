import cv2 as cv
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Read Image
# ----------------------------------------------------------
imgPath = r"C:\Users\dronr\Documents\opencv_project\cats.jpg"

img = cv.imread(imgPath)

if img is None:
    print("Image not found!")
    exit()

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# ----------------------------------------------------------
# Crop Template
# Change these coordinates according to your image if needed
# ----------------------------------------------------------
template = img[80:320, 160:420]

h, w = template.shape[:2]

# ----------------------------------------------------------
# Display Original & Template
# ----------------------------------------------------------
plt.figure(figsize=(10,5))

plt.subplot(121)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(122)
plt.imshow(template)
plt.title("Template")
plt.axis("off")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Matching Methods
# ----------------------------------------------------------
methods = [
    cv.TM_CCOEFF,
    cv.TM_CCOEFF_NORMED,
    cv.TM_CCORR,
    cv.TM_CCORR_NORMED,
    cv.TM_SQDIFF,
    cv.TM_SQDIFF_NORMED
]

titles = [
    "TM_CCOEFF",
    "TM_CCOEFF_NORMED",
    "TM_CCORR",
    "TM_CCORR_NORMED",
    "TM_SQDIFF",
    "TM_SQDIFF_NORMED"
]

# ----------------------------------------------------------
# Apply Every Method
# ----------------------------------------------------------
for method, title in zip(methods, titles):

    currentImage = img.copy()

    # Match Template
    result = cv.matchTemplate(
        currentImage,
        template,
        method
    )

    # Best Match Location
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)

    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        topLeft = minLoc
    else:
        topLeft = maxLoc

    bottomRight = (topLeft[0] + w, topLeft[1] + h)

    # Draw Rectangle
    cv.rectangle(
        currentImage,
        topLeft,
        bottomRight,
        (255, 0, 0),
        3
    )

    # Display Result
    plt.figure(figsize=(12,5))

    plt.subplot(121)
    plt.imshow(result, cmap='gray')
    plt.title(title + " Result Map")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(currentImage)
    plt.title(title + " Detection")
    plt.axis("off")

    plt.tight_layout()

plt.show()