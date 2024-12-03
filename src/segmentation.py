import numpy as np
import cv2
from ultralytics import YOLO
import argparse


def display_webcam():
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

def yolo_webcam():
    model = YOLO('yolov8n.pt')
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        # cv2.imshow('Camera', frame)
        
        results = model(frame)
        # print(results[0].boxes.xywh[0].cpu().tolist())
        # print(results[0].boxes)
        x, y, w, h = results[0].boxes.xywh[0].cpu().tolist()
        # print(results)
        # 
        cv2.rectangle(frame, (int(x) - + int(w/2), int(y) - + int(h/2)), (int(x) + int(w/2), int(y) + int(h/2)), (0, 255, 0), 2)
        # cv2.circle(frame, (int(x), int(y)), 100, (0, 255, 0), 2)
        
        cv2.imshow('Camera', frame)
        

        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


def main(args):
    if args.model == 'none':
        display_webcam()
    elif args.model == 'yolo':
        yolo_webcam()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='none')
    args = parser.parse_args()
    main(args)
