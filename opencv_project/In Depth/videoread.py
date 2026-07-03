import cv2
import os

def videofromcapture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        cv2.imshow('Video Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()  

def videofromfile():
    video_path = r"C:\Users\dronr\Documents\opencv_project\Videos\dog.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        cv2.imshow('Video Frame', frame)
        delay= int(1000/60)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

import cv2 as cv

def writeVideoToFile():
    cap = cv.VideoCapture(0)

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    outPath = r"C:\Users\dronr\Desktop\wao\webcam.avi"

    out = cv.VideoWriter(outPath, fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            out.write(frame)
            cv.imshow('Webcam', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    videofromcapture()
    videofromfile()
    writeVideoToFile()
