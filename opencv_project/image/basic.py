import cv2

# Read image from file
img = cv2.imread('image.jpg')

# Display the image
cv2.imshow('Image', img)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
