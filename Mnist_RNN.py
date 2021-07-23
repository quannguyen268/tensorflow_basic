import numpy as np
import  tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()
import os
from keras.datasets import fashion_mnist

# define parameter
element_size = 28
time_step = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

# Where to save TensorBoard model summaries
LOG_DIR = "logs/RNN_with_summaries"



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



# Create placeholders for inputs, labels
_inputs = tf.placeholder(tf.float32, shape=[None, time_step, element_size], name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')

# Weights and bias for input and hidden layer
with tf.name_scope('rnn_weights'):
    with tf.name_scope("W_x"):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        variable_summaries(Wx)
    with tf.name_scope('W_h'):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
        variable_summaries(Wh)
    with tf.name_scope('Bias'):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)


def RNN_Step(previous_hidden_state, x):
    curren_hidden_state = tf.tanh(
        tf.matmul(previous_hidden_state, Wh) +
        tf.matmul(x, Wx) + b_rnn
    )

    return curren_hidden_state


processed_input = tf.transpose(_inputs, perm=[1, 0, 2]) # (timestep, batch_size, element_size)
initial_hidden = tf.zeros([batch_size, hidden_layer_size])

# Geting all state vector across time
all_hidden_state = tf.scan(RNN_Step, processed_input, initializer=initial_hidden, name='states')


#%%
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

print(y_test.shape)

#%%

# Weights for output layers
with tf.name_scope('linear_layer_weights') as scope:
    with tf.name_scope('W_linear'):
        Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],
                                             mean=0, stddev=.01))
        variable_summaries(Wl)

    with tf.name_scope("Bias_linear"):
        bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))
        variable_summaries(bl)

# Apply linear layer to state vector
def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state, Wl) + bl

with tf.name_scope('linear_layer_weights') as scope:
    #Iterate across time, apply linear layer to all RNN outputs
    all_outputs = tf.map_fn(get_linear_layer, all_hidden_state)
    # Get last output
    output = all_outputs[-1]
    tf.summary.histogram('outputs', output)

with tf.name_scope('cross_entroypy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(output, 1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
    tf.summary.scalar('accuracy', accuracy)


#Merge all the summaries
merged = tf.summary.merge_all()

# Get a small test set
test_data = x_test[:batch_size].reshape((-1, time_step, element_size))
tes_label = y_test[:batch_size]

with tf.Session() as sess:
    #Write summaries to LOG_DIR -- used by Tensorboard
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test', graph=tf.get_default_graph())

    sess.run(tf.global_variables_initializer())
    j = 0
    for i in range(10000):


        x_batch = x_train[j: j + batch_size].reshape((-1, time_step, element_size))
        y_batch = y_train[j: j + batch_size]


        summary, _ = sess.run([merged, train_step], feed_dict={_inputs: x_batch,
                                                              y: y_batch})
        # Add to summaries
        train_writer.add_summary(summary, i)

        if i % 100 ==0:
            acc, loss, = sess.run([accuracy, cross_entropy], feed_dict={_inputs: x_batch,
                                                                        y: y_batch})
            print("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        if i % 10 == 0:
            # add to summary
            summary, acc = sess.run([merged, accuracy], feed_dict={_inputs: test_data,
                                                                   y: tes_label})
            test_writer.add_summary(summary, i)

    test_acc = sess.run(accuracy, feed_dict={_inputs: test_data,
                                             y: tes_label})
    print("Test Accuracy:", test_acc)





#%%
