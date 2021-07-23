import tensorflow as tf
import numpy as np

tf = tf.compat.v1
tf.disable_v2_behavior()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()



y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

x_test = x_test.reshape(10, 1000, 784)
y_test = y_test.reshape(10, 1000, 10)

x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

#%%

with tf.Graph().as_default():
    x_ = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope('CNN_Model'):
        x_image = tf.reshape(x_, [-1, 28, 28, 1])
        conv1 = tf.keras.layers.Conv2D( 32, [5,5], activation=tf.nn.swish)(x_image)
        pool1 = tf.keras.layers.MaxPool2D( [2,2])(conv1)

        conv2 = tf.keras.layers.Conv2D( 64, [5,5],
                                       activation=tf.nn.swish)(pool1)
        pool2 = tf.keras.layers.MaxPool2D( [2,2])(conv2)

        flat = tf.reshape( pool2, [-1, 7*7*64])

        fc1 = tf.keras.layers.Dense( 1024, activation=tf.nn.swish)(flat)
        drop = tf.nn.dropout( fc1, keep_prob=0.5)

        y_conv = tf.keras.layers.Dense( 10, activation=None)(drop)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#%%


