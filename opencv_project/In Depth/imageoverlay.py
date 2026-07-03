import cv2 as cv
import matplotlib.pyplot as plt

def imageOverlay():
    # Read images
    img = cv.imread(r"C:\Users\dronr\Documents\opencv_project\cat.jpg")
    logo = cv.imread(r"C:\Users\dronr\Documents\opencv_project\teslalogo.png")

    if img is None or logo is None:
        print("Error: Could not load one or both images.")
        return
    logo = cv.resize(logo, (150, 150))
    h, w = logo.shape[:2]
    roi = img[0:h, 0:w]
    gray = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    img_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    logo_fg = cv.bitwise_and(logo, logo, mask=mask)
    dst = cv.add(img_bg, logo_fg)
    img[0:h, 0:w] = dst
    result = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(result)
    plt.title("Tesla Logo Overlay on Cat")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    imageOverlay()