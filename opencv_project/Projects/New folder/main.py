import cv2

import config
import utils

from landmarks import LandmarkDetector
from measurements import calculate_measurements
from symmetry import calculate_symmetry
from face_shape import classify_face_shape
from recommendation import recommend_glasses


def main():

    detector = LandmarkDetector(config.PREDICTOR_PATH)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to open webcam.")
        return

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------
        # Detect Face & Landmarks
        # ----------------------------------

        faces, landmark_list = detector.detect(frame)

        for face in faces:
            utils.draw_rectangle(frame, face)

        # ----------------------------------
        # Process every detected face
        # ----------------------------------

        for landmarks in landmark_list:

            # Draw landmarks
            utils.draw_landmarks(frame, landmarks)

            # Draw measurement lines
            utils.draw_measurements(frame, landmarks)

            # Face measurements
            data = calculate_measurements(landmarks)

            # Symmetry
            symmetry, center_x, angle = calculate_symmetry(landmarks)

            utils.draw_symmetry(frame, landmarks, center_x)

            # Face Shape
            face_shape = classify_face_shape(data)

            # Recommendation
            recommendation = recommend_glasses(
                face_shape,
                data["face_width"]
            )

            frame_size = recommendation["size"]

            # ------------------------------
            # Head Orientation
            # ------------------------------

            if abs(angle) < 5:
                orientation = "Straight"

            elif angle > 5:
                orientation = "Tilted Right"

            else:
                orientation = "Tilted Left"

            # ------------------------------
            # Symmetry Color
            # ------------------------------

            if symmetry >= 95:
                sym_color = (0, 255, 0)

            elif symmetry >= 85:
                sym_color = (0, 255, 255)

            else:
                sym_color = (0, 0, 255)

            # ------------------------------
            # Display Information
            # ------------------------------

            y = 30

            info = [

                ("Face Width", f"{data['face_width']:.1f}px"),

                ("Face Height", f"{data['face_height']:.1f}px"),

                ("Jaw Width", f"{data['jaw_width']:.1f}px"),

                ("Forehead Width", f"{data['forehead_width']:.1f}px"),

                ("Eye Distance", f"{data['eye_distance']:.1f}px"),

                ("Nose Width", f"{data['nose_width']:.1f}px"),

                ("Mouth Width", f"{data['mouth_width']:.1f}px"),

                ("Face Ratio", f"{data['face_ratio']:.2f}")

            ]

            for label, value in info:

                cv2.putText(
                    frame,
                    f"{label} : {value}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                y += 30

            cv2.putText(
                frame,
                f"Symmetry : {symmetry:.2f} %",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                sym_color,
                2
            )

            y += 30

            cv2.putText(
                frame,
                f"Tilt Angle : {angle:.2f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            y += 30

            cv2.putText(
                frame,
                f"Orientation : {orientation}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )

            y += 30

            cv2.putText(
                frame,
                f"Face Shape : {face_shape}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            y += 30

            cv2.putText(
                frame,
                f"Frame Size : {frame_size}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            y += 30

            cv2.putText(
                frame,
                "Recommended Glasses:",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            y += 30

            for frame_name in recommendation["frames"]:

                cv2.putText(
                    frame,
                    f"- {frame_name}",
                    (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2
                )

                y += 25

        # ----------------------------------
        # Display Window
        # ----------------------------------

        cv2.imshow("AI Face Analysis & Glasses Recommendation", frame)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()