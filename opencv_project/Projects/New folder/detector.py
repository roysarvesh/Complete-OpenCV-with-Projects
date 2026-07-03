"""
Face Detection Module
"""

import cv2
import config


class FaceDetector:

    def __init__(self):

        self.face_detector = cv2.CascadeClassifier(
            config.CASCADE_PATH
        )

    def detect(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=config.SCALE_FACTOR,
            minNeighbors=config.MIN_NEIGHBORS,
            minSize=config.MIN_SIZE,
        )

        return faces