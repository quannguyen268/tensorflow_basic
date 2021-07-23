import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

tf = tf.compat.v1
tf.disable_v2_behavior()
file_name = "/home/quan/PycharmProjects/tensorflow_basic/data/cifar10.tfrecords"

# Download data to save_dir
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x = np.vstack((x_train, x_test))
y = np.vstack((y_train, y_test))

#%%

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



writer = tf.python_io.TFRecordWriter(file_name)
for index in range(x.shape[0]):
    image = x[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(x.shape[1]),
        'width' :_int64_feature(x.shape[2]),
        'depth' : _int64_feature(x.shape[3]),
        'label' : _int64_feature(int(y[index])),
        'image_raw':_bytes_feature(image),

    }))
writer.write(example.SerializeToString())
writer.close()

#%%

record_iterator = tf.python_io.tf_record_iterator(file_name)
serialize_img_example = next(record_iterator)

example = tf.train.Example()
example.ParseFromString(serialize_img_example)
image = example.features.feature['image_raw'].bytes_list.value
label = example.features.feature['label'].int64_list.value[0]
width = example.features.feature['width'].int64_list.value[0]

print(width)

#%%
import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()

tf.nn.conv2d
tf.keras.layers.Dense
