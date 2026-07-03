import cv2
import numpy as np
img = np.zeros((500, 500, 3), dtype=np.uint8)
cv2.line(img, (50, 50), (450, 50), (255, 255, 255), 3)
cv2.rectangle(img, (100, 100), (300, 250), (0, 255, 0), 3)
cv2.circle(img, (375, 350), 60, (0, 0, 255), 3)
cv2.imshow("Shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
