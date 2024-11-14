import numpy as np
import cv2

def display_webcam():
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

def main():
    display_webcam()
    pass


if __name__ == "__main__":
    main()
