# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import, division, print_function

import argparse
import functools
from functools import partial
import os

import tensorflow as tf

NUM_CLASSES = 10

BATCH_SIZE = 64
IMAGE_SIZE = [32, 32]

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {"image": tf.io.FixedLenFeature([], tf.string),}
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled:
        label = tf.cast(example["label"], tf.int32)
        return image, label
    return image


def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=BATCH_SIZE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

def get_dataset(filenames, labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    dataset = dataset.map(normalize_img)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=BATCH_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

def train(model_dir, data_dir, train_steps):
    
    ds_train = get_dataset(os.path.join(data_dir, "train.tfrecords"))
    ds_test = get_dataset(os.path.join(data_dir, "eval.tfrecords"))
        
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation = 'softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    
    model.fit(ds_train, epochs=train_steps, validation_data=ds_test)
    
    scores = model.evaluate(ds_test, verbose=2)
    print(
        "Validation results: "
        + "; ".join(map(
            lambda i: f"{model.metrics_names[i]}={scores[i]:.5f}", range(len(model.metrics_names))
        ))
    )

    ###### Save Keras model for TensorFlow Serving ############
    export_path = f"{model_dir}/1"

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True
    )

def main(model_dir, data_dir, train_steps):
    train(model_dir, data_dir, train_steps)


if __name__ == "__main__":
    
    args_parser = argparse.ArgumentParser()
    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    args_parser.add_argument(
        "--data-dir",
        default='/opt/ml/input/data/training',
        type=str,
        help="The directory where the CIFAR-10 input data is stored. Default: /opt/ml/input/data/training. This "
        "directory corresponds to the SageMaker channel named 'training', which was specified when creating "
        "our training job on SageMaker",
    )
    
    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html
    args_parser.add_argument(
        "--model-dir",
        default="/opt/ml/model",
        type=str,
        help="The directory where the model will be stored. Default: /opt/ml/model. This directory should contain all "
        "final model artifacts as Amazon SageMaker copies all data within this directory as a single object in "
        "compressed tar format.",
    )

    args_parser.add_argument(
        "--train-steps", type=int, default=20, help="The number of steps to use for training."
    )
    args = args_parser.parse_args()
    main(**vars(args))
