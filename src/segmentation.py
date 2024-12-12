import numpy as np
import cv2
from ultralytics import YOLO
import argparse
import time
from unet.videoprocess import preprocess_frame

def calculate_fps(start_time, frame_count):
    current_time = time.time()
    fps = frame_count / (current_time - start_time)
    return fps

def display_webcam():
    cam = cv2.VideoCapture(0)
    print(cv2.VideoCapture().getBackendName())
    if cam.isOpened():
        print("Camera loaded successfully!")
    else:
        print("Failed to load camera.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("无法捕获帧。")
            break
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

def unet_webcam():
    cam = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time()
    while True:
        ret, frame = cam.read()
        # get mask from unet model
        mask_resized = preprocess_frame(frame)

        # apply mask
        frame[mask_resized == 0] = 0  # black background

        # show result
        cv2.imshow("U-Net Processed Camera", frame)

        # show frame per second
        frame_count += 1
        fps = calculate_fps(start_time, frame_count)
        print(f"FPS: {fps:.2f}")

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def main(args):
    if args.model == 'none':
        display_webcam()
    elif args.model == 'yolo':
        yolo_webcam()
    elif args.model == 'unet':
        unet_webcam()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='none')
    args = parser.parse_args()
    main(args)
