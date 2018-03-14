'''
File: save_to_bytes.py

This file is a small helper script to convert all the images into a byte stream

Assumes a folder called train/ and a folder called eval/ within the working directory
'''

import pickle
import os 

import numpy as np
import tensorflow as tf

TRAIN_PATH = './training_reduced'
EVAL_PATH = './evaluation_reduced'

def main():
    train_filenames = [os.path.join(TRAIN_PATH, f) for f in os.listdir(TRAIN_PATH)
                       if f.endswith('.jpg')]
    eval_filenames = [os.path.join(EVAL_PATH, f) for f in os.listdir(EVAL_PATH)
                       if f.endswith('.jpg')]

    train = load_data(train_filenames)
    evaluate = load_data(eval_filenames)

    save(train, "train_images")
    save(evaluate, "eval_images")


def save(arr, filename):
    with open(filename, "wb") as file:
        pickle.dump(arr, file)


def load_data(filenames):
    parse_fn = lambda f: _parse_function(f)
    images = tf.constant(filenames)
    processed = tf.map_fn(parse_fn, images, dtype=tf.float32)
    with tf.Session() as sess:
        data = sess.run(processed)
    return data


def _parse_function(filename, size=180):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize_images(image, [size, size])

    return resized_image


if __name__ == '__main__':
    main()