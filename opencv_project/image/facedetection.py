import cv2

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Read the image
img = cv2.imread("man.jpeg")

if img is None:
    print("Error: man.jpeg not found!")
    exit()

# Convert image to grayscale
# Haar Cascades work only on grayscale images.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ----------------------------------------------------
# Detect faces in the image
# scaleFactor -> Reduces image size at each scale.
# minNeighbors -> Higher value reduces false detections.
# minSize -> Minimum face size to detect.
# ----------------------------------------------------
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print("Number of faces detected:", len(faces))

# ----------------------------------------------------
# Draw a green rectangle around each detected face
# ----------------------------------------------------
for (x, y, w, h) in faces:
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

# Display the result
cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()