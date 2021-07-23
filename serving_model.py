import tensorflow as tf
import os
import sys
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants, signature_def_utils, tag_constants, utils
from tensorflow.python.util import compat
from tensorflow.keras.datasets import mnist
tf = tf.compat.v1
tf.disable_v2_behavior()


tf.app.flags.DEFINE_integer('training_iteration', 10, 'number of training iteration')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/home/quan/PycharmProjects/tensorflow_basic/logs', 'Working directory')

FLAGS = tf.app.flags.FLAGS

class basic_cnn:
    def __init__(self, x_image, keep_prop, weights=None, sess=None):
        self.parameters = []
        self.x_image = x_image

        # conv1 = sef.conv_layer


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name='weights')

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name='weights')

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='same')

    def maxpool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1])

    def conv_layer(self, input, shape):
        W = self.weight_variable(shape)
        b = self.bias_variable(shape[3])
        return tf.nn.swish(self.conv2d(input, W) + b)

    def full_layer(self, input, size):
        in_size = int(input.get_shape()[1])
        W = self.weight_variable([in_size, size])
        b = self.bias_variable([size])
        self.parameters += [W,b]
        return tf.nn.swish(tf.matmul(input, W) + b)

    def load_weights(self, weights, sess):
        for i,w in enumerate(weights):
            print("Weights index: {}".format(i), "Weights shape: {}".format(w.shape))
            sess.run(self.parameters[i].assign(w))

    def main(self,_):
        if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
            print('Usage: mnist_export.py [--training_iteration=x] '
                                    '[--model_version=y] export_dir')

        if FLAGS.training_iteration <= 0:
            print('please specify a positive value for training iteration.')
            sys.exit(-1)

        if FLAGS.model_version <= 0:
            print('please specify a positive value for version number.')
            sys.exit(-1)

        print('Training...')

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        with tf.Session() as sess:
            serialize_tf_example = tf.placeholder(tf.string, name='tf_example')
            feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32)}
            tf_example = tf.parse_example(serialize_tf_example, feature_configs)

            x = tf.identity(tf_example['x'], name='x')
            y_ = tf.placeholder('float', shape=[None, 10])


            x_image = tf.reshape(x, [-1, 28, 28, 1])
            h_conv1 = self.conv_layer(x_image, [5, 5, 1, 32])
            h_pool1 = self.maxpool_2x2(h_conv1)

            h_conv2 = self.conv_layer(h_pool1, [5, 5, 32, 64])
            h_pool2 = self.maxpool_2x2(h_conv2)

            h_fc1 = self.full_layer(h_pool2, [-1, 7*7*64])

            keep_prop = tf.placeholder(tf.float32)
            h_drop1 = tf.nn.dropout(h_fc1, keep_prop)

            w_fc2 = self.weight_variable([1024, 10])
            b_fc2 = self.bias_variable([10])

            y_conv = tf.matmul(h_drop1,w_fc2) + b_fc2

            y = tf.nn.softmax(y_conv, name='y')
            cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

            values, indices = tf.nn.top_k(y_conv, 10)

            for _ in range(FLAGS.training_iteration):
                for i in range(50):
                    for j in range(0, x_train.shape[0], 128):
                        x_batch = x_train[j: j + 128]
                        y_batch = y_train[j: j + 128]

                        train_step.run(feed_dict={x: x_batch,
                                                  y_: y_batch, keep_prop: 0.5})
                        print(_)
                        correct_prediction = tf.equal(tf.argmax(y_conv, 1),
                                                      tf.argmax(y_, 1))
                        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
                        print('training accuracy %g' % accuracy.eval(feed_dict={
                            x: mnist.test.images,
                            y_: mnist.test.labels, keep_prop: 1.0}))
                        print('training is finished!')






