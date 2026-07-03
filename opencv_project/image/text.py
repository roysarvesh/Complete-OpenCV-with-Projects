import cv2
import numpy as np
img = np.zeros((500, 700, 3), dtype=np.uint8)
cv2.putText(
    img,                    # Image
    "Hello, My name is Sarvesh!",       # Text
    (100, 250),             # Bottom-left corner of text
    cv2.FONT_HERSHEY_SIMPLEX, # Font
    1.5,                    # Font scale
    (0, 255, 255),          # Color (B, G, R)
    3,                      # Thickness
    cv2.LINE_AA             # Line type (Anti-aliased)
)
cv2.imshow("Text on Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()