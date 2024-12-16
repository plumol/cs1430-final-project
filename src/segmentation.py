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
           # Check if any objects were detected
        if results[0].boxes:  
            # Extract bounding box information
            for box in results[0].boxes:
                x, y, w, h = box.xywh.cpu().numpy()[0]
                # Draw bounding box
                cv2.rectangle(frame, 
                              (int(x - w / 2), int(y - h / 2)), 
                              (int(x + w / 2), int(y + h / 2)), 
                              (0, 255, 0), 2)
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

def unet_webcam():
    cam = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time()

    default_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    default_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # background image
    back_image= cv2.imread("./data/background.png")
    back_image = cv2.resize(back_image, (default_width, default_height))
    background = back_image.copy()

    skip_frames = 3  
    #cache the mask
    mask_resized = None  
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            # recaculate mask after skip_frames 
            mask_resized = preprocess_frame(frame)

         # apply changing mask
        if mask_resized is not None:
            frame[mask_resized == 0] = 0  
            background = back_image.copy()
            background[mask_resized == 1] = 0
         # show result
        cv2.imshow("U-Net Processed Camera", frame + background)

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