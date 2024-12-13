import cv2
import numpy as np  
import tensorflow as tf
import os
from keras.models import load_model
import tensorflow.lite as tflite
from .hyperparameters import img_size

def iou_loss(y_true, y_pred):
    epsilon = 1e-7
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    iou_loss = 1 - iou
    # Average IoU loss across the batch
    return tf.reduce_mean(iou_loss)

# Combined loss: binary_crossentropy + IoU loss
def combined_loss(y_true, y_pred):
    bce =  tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    iou = iou_loss(y_true, y_pred)
    return bce + iou  


# save TFLite model
if not os.path.exists("unet_model.tflite"):
    model = load_model("best_model.h5", custom_objects={
    "iou_loss": iou_loss,
    "combined_loss": combined_loss
    })
    # transfer to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open("unet_model.tflite", "wb") as f:
        f.write(tflite_model)
#experimental_delegates = [tf.lite.experimental.load_delegate('libOpenCL.so')]
interpreter = tflite.Interpreter(model_path="unet_model.tflite", 
                                num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_frame(frame, img_size = img_size):
    #transform frame to U-Net accepted image
    frame_resized = cv2.resize(frame, (img_size, img_size))  # resize
    frame_input = np.expand_dims(frame_resized.astype(np.float32) / 255.0, axis=0)  # increase batch demension
    
    # get mask 
    interpreter.set_tensor(input_details[0]['index'], frame_input)
    interpreter.invoke()
    mask = interpreter.get_tensor(output_details[0]['index'])[0]
    mask_binary = (mask > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask_binary, (frame.shape[1], frame.shape[0])) 
    return mask_resized

