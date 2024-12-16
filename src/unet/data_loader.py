import os
import tensorflow as tf
import cv2
import numpy as np
import hyperparameters as hp

def generate_file_paths(img_folder, lbl_folder):
    img_files = sorted(os.listdir(img_folder))
    lbl_files = sorted(os.listdir(lbl_folder))
    assert set(img_files) == set(lbl_files), "Mismatch between image and label filenames."

    img_paths = [os.path.join(img_folder, f) for f in img_files]
    lbl_paths = [os.path.join(lbl_folder, f) for f in img_files]
    return img_paths, lbl_paths

def load_and_preprocess_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (hp.img_size, hp.img_size))
    image = image.astype(np.float32) / 255.0
    return image

def load_and_process_label(filepath):
    label = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label, (hp.img_size, hp.img_size))
    label = (label / 255.0) > 0.5
    label = label.astype(np.float32)[..., np.newaxis]
    return label

def pipeline_preprocessing(img_path, lbl_path):
    def inner_function(img_path, lbl_path):
        img_path = img_path.decode("utf-8")
        lbl_path = lbl_path.decode("utf-8")

        img = load_and_preprocess_image(img_path)
        lbl = load_and_process_label(lbl_path)

        return img, lbl

    img_tensor, lbl_tensor = tf.numpy_function(inner_function, [img_path, lbl_path], [tf.float32, tf.float32])
    img_tensor.set_shape([hp.img_size, hp.img_size, 3])
    lbl_tensor.set_shape([hp.img_size, hp.img_size, 1])

    return img_tensor, lbl_tensor

def create_dataset_pipeline(image_list, label_list, batch_size):
    data = tf.data.Dataset.from_tensor_slices((image_list, label_list))
    data = data.shuffle(buffer_size=5000)
    data = data.map(pipeline_preprocessing)
    data = data.batch(batch_size)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data
