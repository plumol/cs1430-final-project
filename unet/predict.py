from models import unet_model
import hyperparameters as hp
import os
import numpy as np
import cv2
from data_loader import read_image  

def predict_and_save(model_path, image_folder, output_folder):
    # load best model
    model = unet_model(input_shape=(hp.img_size, hp.img_size, 3))
    model.load_weights(model_path)

    # create predict folder
    os.makedirs(output_folder, exist_ok=True)

    # get all images
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg'))]

    for image_path in image_paths:
        # preprocess
        image = read_image(image_path) 

        # Shape: (1, 256, 256, 3)
        image_batch = np.expand_dims(image, axis=0)  
        prediction = model.predict(image_batch)  

        # Shape: (256, 256, 1)
        prediction = np.squeeze(prediction, axis=0)  

        # 0 or 1
        prediction_binary = (prediction > 0.5).astype(np.uint8)  

        # grayscale
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, prediction_binary * 255) 

        print(f"Saved segmentation result to {output_path}")

if __name__ == "__main__":
    model_path = "best_model.h5"
    image_folder = "../data/images/"
    output_folder = "./predict/"

    predict_and_save(model_path, image_folder, output_folder)
