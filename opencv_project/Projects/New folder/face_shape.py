"""
face_shape.py

Face Shape Classification
"""


def classify_face_shape(data):
    """
    data -> dictionary returned by calculate_measurements()

    Returns:
        face_shape
    """

    fw = data["face_width"]
    fh = data["face_height"]
    jw = data["jaw_width"]
    fwhead = data["forehead_width"]
    ratio = data["face_ratio"]

    # Height approximately equals width
    if ratio < 1.10:

        if abs(jw - fw) < 20:
            return "Round"

        return "Square"

    # Long face
    if ratio > 1.45:
        return "Rectangle"

    # Forehead wider than jaw
    if fwhead > jw + 20:
        return "Heart"

    # Jaw much narrower than cheeks
    if jw < fw * 0.75:
        return "Diamond"

    return "Oval"