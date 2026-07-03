import cv2
import dlib
import numpy as np

class LandmarkDetector:

    def __init__(self, predictor_path):

        self.detector = dlib.get_frontal_face_detector()

        self.predictor = dlib.shape_predictor(
            predictor_path
        )

    def detect(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray)

        landmarks = []

        for face in faces:

            shape = self.predictor(gray, face)

            pts = []

            for i in range(68):

                x = shape.part(i).x
                y = shape.part(i).y

                pts.append((x, y))

            landmarks.append(np.array(pts))

        return faces, landmarks