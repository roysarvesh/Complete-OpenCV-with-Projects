"""
symmetry.py

Facial Symmetry Analysis using dlib 68 landmarks
"""

import math
import cv2


def euclidean(p1, p2):
    return math.sqrt(
        (p1[0]-p2[0])**2 +
        (p1[1]-p2[1])**2
    )


def calculate_symmetry(landmarks):
    """
    Returns:
        symmetry percentage
        face center x
        tilt angle
    """

    # Face center
    center_x = int((landmarks[0][0] + landmarks[16][0]) / 2)

    # Landmark pairs
    pairs = [
        (0,16),
        (1,15),
        (2,14),
        (3,13),
        (4,12),
        (5,11),
        (6,10),
        (7,9),

        (17,26),
        (18,25),
        (19,24),
        (20,23),
        (21,22),

        (36,45),
        (37,44),
        (38,43),
        (39,42),

        (31,35),
        (48,54)
    ]

    total_error = 0

    for left,right in pairs:

        left_distance = abs(center_x - landmarks[left][0])
        right_distance = abs(landmarks[right][0] - center_x)

        total_error += abs(left_distance-right_distance)

    average_error = total_error/len(pairs)

    face_width = abs(landmarks[16][0]-landmarks[0][0])

    symmetry = 100 - (average_error/face_width)*100

    symmetry = max(0,min(symmetry,100))

    # Face tilt

    left_eye = (
        (landmarks[36][0]+landmarks[39][0])/2,
        (landmarks[36][1]+landmarks[39][1])/2
    )

    right_eye = (
        (landmarks[42][0]+landmarks[45][0])/2,
        (landmarks[42][1]+landmarks[45][1])/2
    )

    dx = right_eye[0]-left_eye[0]
    dy = right_eye[1]-left_eye[1]

    angle = math.degrees(math.atan2(dy,dx))

    return symmetry, center_x, angle