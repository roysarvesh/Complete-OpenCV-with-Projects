import cv2 as cv
import numpy as np
blank = np.zeros((400, 400), dtype='uint8')
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)
cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)
# --------------------------------------------------
# Bitwise AND --> Intersecting (common) regions
# Displays only the overlapping area of the rectangle and circle.
# --------------------------------------------------
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise AND', bitwise_and)

# --------------------------------------------------
# Bitwise OR --> Non-intersecting and intersecting regions
# Displays all white regions from both images.
# --------------------------------------------------
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('Bitwise OR', bitwise_or)

# --------------------------------------------------
# Bitwise XOR --> Non-intersecting regions
# Displays only the regions that do NOT overlap.
# --------------------------------------------------
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('Bitwise XOR', bitwise_xor)

# --------------------------------------------------
# Bitwise NOT --> Inverts the image
# Converts white pixels to black and black pixels to white.
# --------------------------------------------------
bitwise_not = cv.bitwise_not(rectangle)
cv.imshow('Bitwise NOT', bitwise_not)

# Wait until a key is pressed
cv.waitKey(0)

# Close all OpenCV windows
cv.destroyAllWindows()