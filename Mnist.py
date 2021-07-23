import random

import tensorflow as tf
import numpy as np
import sklearn
tf = tf.compat.v1
tf.disable_eager_execution()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, s):
    return tf.nn.max_pool(x, ksize=[1, s, s, 1], strides=[1 ,s, s, 1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b

x = tf.placeholder(tf.float32, shape=[None, 28,28,1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])
conv1 = conv_layer(x_image, shape=[5, 5, 1, 32]) # 28x28x32
pool1 = max_pool(conv1, 2) # 14x14x32

conv2 = conv_layer(pool1, shape=[5, 5, 32, 64]) # 14x14x64
pool2 = max_pool(conv2, 2)

flat1 = tf.reshape(pool2, [-1, 7*7*64])
full1 = tf.nn.relu(full_layer(flat1, 1024))
keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)




#%%

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()



y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

x_test = x_test.reshape(10, 1000, 784)
y_test = y_test.reshape(10, 1000, 10)

x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

#%%

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6, allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(50):
        for j in range(0,x_train.shape[0],128):
            x_batch = x_train[j: j + 128]
            y_batch = y_train[j: j + 128]

            if i % 10 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: x_batch,y_:y_batch,keep_prob: 1.0})

                print("step {}, training accuracy {}".format(i, train_accuracy))

            sess.run(train_step, feed_dict={x: x_batch, y_: y_batch, keep_prob:0.75})



        test_acc = np.mean([sess.run(accuracy, feed_dict={x: x_test[i],y_: y_test[i],keep_prob: 1.0})
                            for i in range(10)])

        print("test accuracy: {}".format(test_acc))


#%%
