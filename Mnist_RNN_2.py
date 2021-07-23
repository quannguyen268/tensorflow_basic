import numpy as np
import  tensorflow as tf
import os
from keras.datasets import fashion_mnist

# define parameter
element_size = 28
time_step = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

# Where to save TensorBoard model summaries
LOG_DIR = "logs/Sumary_2"

# Add some ops that take care of logging summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


x_ = tf.placeholder(tf.float32, shape=[None, time_step, element_size], name='inputs')
y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='input')

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#%%
# Tensorflow built-in functions
rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_layer_size)
outputs, _ = tf.nn.dynamic_rnn(rnn_cell, x_, dtype=tf.float32)

# Weights for output layers
with tf.name_scope('linear_layer_weights') as scope:
    with tf.name_scope('W_linear'):
        Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],
                                             mean=0, stddev=.01))
        variable_summaries(Wl)

    with tf.name_scope("Bias_linear"):
        bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))
        variable_summaries(bl)


def get_linear_layer(vector):
    return tf.matmul(vector, Wl) + bl

with tf.name_scope('linear_layer_weights') as scope:
    last_rnn_output = outputs[:, -1, :]
    final_output = get_linear_layer(last_rnn_output)
    tf.summary.histogram('outputs', final_output)



with tf.name_scope('cross_entroypy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=y_))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(final_output, 1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
    tf.summary.scalar('accuracy', accuracy)



#Merge all the summaries
merged = tf.summary.merge_all()

#Write summaries to LOG_DIR -- used by Tensorboard
train_writer = tf.summary.FileWriter(LOG_DIR + '/train', graph=tf.get_default_graph())
test_writer = tf.summary.FileWriter(LOG_DIR + '/test', graph=tf.get_default_graph())


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
sess.run(tf.global_variables_initializer())

test_data = x_test[:1000].reshape(-1, time_step, element_size)
test_label = y_test[:1000]
j = 0
for i in range(10000):
    x_batch = x_train[j: j + batch_size].reshape((-1, time_step, element_size))
    y_batch = y_train[j: j + batch_size]
    j = (j + batch_size) % len(x_train)
    if (j + batch_size) >= len(x_train):
        j = j + batch_size - len(x_train)

    summary, _ = sess.run([merged, train_step], feed_dict={x_: x_batch,
                                                           y_: y_batch})
    # Add to summaries
    train_writer.add_summary(summary, i)

    if i % 1000 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict={x_: test_data,
                                                               y_: test_label})
        test_writer.add_summary(summary, i)
        loss = sess.run(cross_entropy, feed_dict={x_: x_batch, y_: y_batch})

        print("Iter " + str(i) + ", Minibatch Loss = " \
              "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

print("Testing Accuracy:", sess.run(accuracy, feed_dict={x_: test_data, y_ : test_label}))