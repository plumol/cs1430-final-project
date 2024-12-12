import os
import tensorflow as tf
import cv2
import numpy as np
import hyperparameters as hp

def get_image_mask_paths(image_dir, mask_dir):
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))
    assert set(image_filenames) == set(mask_filenames), "Image and mask filenames do not match."

    image_paths = [os.path.join(image_dir, fname) for fname in image_filenames]
    mask_paths = [os.path.join(mask_dir, fname) for fname in image_filenames]
    return image_paths, mask_paths

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (hp.img_size, hp.img_size))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (hp.img_size, hp.img_size))
    x = x/255.0
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def preprocess(image_path, mask_path):
    def f(image_path, mask_path):
        image_path = image_path.decode()
        mask_path = mask_path.decode()

        x = read_image(image_path)  
        y = read_mask(mask_path)    

        return x, y

    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([hp.img_size, hp.img_size, 3])
    mask.set_shape([hp.img_size, hp.img_size, 1])

    return image, mask

def tf_dataset(images, masks, batch):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset