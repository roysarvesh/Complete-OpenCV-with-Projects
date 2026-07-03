"""
recommendation.py

Glasses Recommendation
"""


def recommend_glasses(face_shape, face_width):
    """
    Returns recommended frame styles and frame size.

    Parameters:
        face_shape (str): Detected face shape.
        face_width (float): Face width in pixels.

    Returns:
        dict: Recommended frames and size.
    """

    recommendations = {
        "Oval": {
            "frames": [
                "Wayfarer",
                "Rectangle",
                "Aviator"
            ]
        },

        "Round": {
            "frames": [
                "Rectangle",
                "Square",
                "Wayfarer"
            ]
        },

        "Square": {
            "frames": [
                "Round",
                "Oval",
                "Rimless"
            ]
        },

        "Rectangle": {
            "frames": [
                "Large Round",
                "Oversized",
                "Wayfarer"
            ]
        },

        "Heart": {
            "frames": [
                "Rimless",
                "Oval",
                "Aviator"
            ]
        },

        "Diamond": {
            "frames": [
                "Cat Eye",
                "Oval",
                "Rimless"
            ]
        }
    }

    # Determine frame size
    if face_width < 160:
        frame_size = "Small"
    elif face_width < 220:
        frame_size = "Medium"
    else:
        frame_size = "Large"

    # Get frame recommendations
    result = recommendations.get(
        face_shape,
        {"frames": ["Universal"]}
    )

    # Add size to the result
    result["size"] = frame_size

    return result