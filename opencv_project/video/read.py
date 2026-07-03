import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to open camera")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow('Video Capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
