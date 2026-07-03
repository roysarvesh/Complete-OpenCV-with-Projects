"""
utils.py

Utility functions for the Face Glasses Recommendation Project
"""

import cv2
import math


# -------------------------------
# Text
# -------------------------------

def put_text(img, text, position, color=(255, 255, 255), scale=0.6):

    cv2.putText(
        img,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


# -------------------------------
# Face Rectangle
# -------------------------------

def draw_rectangle(img, face, color=(0, 255, 0)):

    # Works for both Haar Cascade and dlib rectangle

    if hasattr(face, "left"):      # dlib rectangle

        x = face.left()
        y = face.top()
        w = face.width()
        h = face.height()

    else:                          # Haar Cascade

        x, y, w, h = face

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        color,
        2
    )


# -------------------------------
# Face Center Line
# -------------------------------

def draw_center_line(img, face):

    if hasattr(face, "left"):

        x = face.left()
        y = face.top()
        w = face.width()
        h = face.height()

    else:

        x, y, w, h = face

    center_x = x + w // 2

    cv2.line(
        img,
        (center_x, y),
        (center_x, y + h),
        (255, 0, 0),
        2
    )


# -------------------------------
# Euclidean Distance
# -------------------------------

def euclidean(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# -------------------------------
# Draw Single Point
# -------------------------------

def draw_point(img, point, color=(0, 255, 255), radius=3):

    cv2.circle(
        img,
        tuple(point),
        radius,
        color,
        -1
    )


# -------------------------------
# Draw All 68 Landmarks
# -------------------------------

def draw_landmarks(frame, landmarks):

    for (x, y) in landmarks:

        cv2.circle(
            frame,
            (int(x), int(y)),
            2,
            (0, 255, 255),
            -1
        )


# -------------------------------
# Draw Facial Measurements
# -------------------------------

def draw_measurements(frame, lm):

    # Face Width
    cv2.line(
        frame,
        tuple(lm[0]),
        tuple(lm[16]),
        (0, 255, 0),
        2
    )

    # Face Height
    cv2.line(
        frame,
        tuple(lm[27]),
        tuple(lm[8]),
        (255, 0, 0),
        2
    )

    # Jaw Width
    cv2.line(
        frame,
        tuple(lm[4]),
        tuple(lm[12]),
        (0, 255, 255),
        2
    )

    # Forehead Width (Estimated)
    cv2.line(
        frame,
        tuple(lm[17]),
        tuple(lm[26]),
        (255, 255, 0),
        2
    )

    # Eye Distance

    left_eye = (
        int((lm[36][0] + lm[39][0]) / 2),
        int((lm[36][1] + lm[39][1]) / 2)
    )

    right_eye = (
        int((lm[42][0] + lm[45][0]) / 2),
        int((lm[42][1] + lm[45][1]) / 2)
    )

    cv2.line(
        frame,
        left_eye,
        right_eye,
        (255, 0, 255),
        2
    )

    # Draw Centers

    cv2.circle(frame, left_eye, 4, (0, 0, 255), -1)
    cv2.circle(frame, right_eye, 4, (0, 0, 255), -1)
def draw_symmetry(frame, landmarks, center_x):

    top = landmarks[19][1]

    bottom = landmarks[8][1]

    cv2.line(
        frame,
        (center_x, top),
        (center_x, bottom),
        (0,0,255),
        2
    )