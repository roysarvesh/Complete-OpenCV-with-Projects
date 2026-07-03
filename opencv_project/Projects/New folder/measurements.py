import math

# ---------- Utility Function ----------

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# ---------- Facial Measurements ----------

def calculate_measurements(landmarks):
    """
    landmarks: numpy array of shape (68,2)
    returns dictionary of facial measurements
    """

    # Face Width (Jaw Left -> Jaw Right)
    face_width = distance(landmarks[0], landmarks[16])

    # Face Height (Eyebrow Center -> Chin)
    face_height = distance(landmarks[27], landmarks[8])

    # Jaw Width
    jaw_width = distance(landmarks[4], landmarks[12])

    # Forehead Width (Estimated using eyebrows)
    forehead_width = distance(landmarks[17], landmarks[26])

    # Eye Distance
    left_eye_center = (
        int((landmarks[36][0] + landmarks[39][0]) / 2),
        int((landmarks[36][1] + landmarks[39][1]) / 2),
    )

    right_eye_center = (
        int((landmarks[42][0] + landmarks[45][0]) / 2),
        int((landmarks[42][1] + landmarks[45][1]) / 2),
    )

    eye_distance = distance(left_eye_center, right_eye_center)

    # Nose Width
    nose_width = distance(landmarks[31], landmarks[35])

    # Mouth Width
    mouth_width = distance(landmarks[48], landmarks[54])

    # Face Ratio
    face_ratio = face_height / face_width if face_width != 0 else 0

    return {
        "face_width": face_width,
        "face_height": face_height,
        "jaw_width": jaw_width,
        "forehead_width": forehead_width,
        "eye_distance": eye_distance,
        "nose_width": nose_width,
        "mouth_width": mouth_width,
        "face_ratio": face_ratio,
        "left_eye_center": left_eye_center,
        "right_eye_center": right_eye_center,
    }