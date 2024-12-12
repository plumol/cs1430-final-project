import cv2
import numpy as np  
import tensorflow as tf
from keras.models import load_model
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

model = load_model("best_model.h5", custom_objects={
    "iou_loss": iou_loss,
    "combined_loss": combined_loss
})

def preprocess_frame(frame, img_size = img_size):
    #transform frame to U-Net accepted image
    frame_resized = cv2.resize(frame, (img_size, img_size))  # resize
    frame_normalized = frame_resized / 255.0  # normalize to [0, 1]
    frame_input = np.expand_dims(frame_normalized, axis=0)  # increase batch demension
    
    # get mask 
    mask = model.predict(frame_input)[0]
    mask_binary = (mask > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask_binary, (frame.shape[1], frame.shape[0])) 
    return mask_resized

