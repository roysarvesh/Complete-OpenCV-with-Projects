import cv2
CAMERA_ID = 0
SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 5
MIN_SIZE = (120, 120)

# Colors (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Window
WINDOW_NAME = "AI Face & Glasses Recommendation"

# Haar Cascade
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
PREDICTOR_PATH = r"C:\Users\dronr\Documents\opencv_project\Projects\New folder\shape_predictor_68_face_landmarks.txt"