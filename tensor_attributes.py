import tensorflow as tf
import numpy as np

c = tf.constant([[1, 2, 3],
                 [4, 5, 6]])
print("Python List input: {}".format(c.get_shape()))

c = tf.constant(np.array([
    [[1, 2, 3],
     [4, 5, 6]],
    [[1, 1, 1],
     [2, 2, 2]]
]))

print("3d Numpy array input: {}".format(c.get_shape()))

a = tf.constant(3)
b = tf.constant(5)
c = tf.add(a, b)
d = tf.multiply(a, b)
a = tf.subtract(c, d)

sess = tf.InteractiveSession()
fetch = [a, b, c, d]
# noinspection PyRedeclaration
fetch = tf.truncated_normal((4, 4), 0, 3)
outs = sess.run(fetch)

print("outs = {}".format(outs))

# %%

a = tf.constant([[1, 2, 3],
                 [4, 5, 6]])

x = tf.constant([1, 0, 1])

x = tf.expand_dims(x, 1)

b = tf.matmul(a, x)

b = tf.transpose(b, conjugate=False)
sess = tf.InteractiveSession()
out = sess.run(b)
print("matmul result: \n {}".format(out))

# %%

with tf.Graph().as_default():
    c1 = tf.constant(4, dtype=tf.float16, name='c')
    c2 = tf.constant(4, name='c', dtype=tf.float16)
    with tf.name_scope("prefix"):
        c3 = tf.constant(4, dtype=tf.int8, name='c')
        c4 = tf.constant(4, name='c')
print(c1.name)
print(c2.name)
print(c3.name)
print(c4.name)

# %%

init_val = tf.random.normal((1, 5), 0, 1)
var = tf.Variable(init_val, name='var')
print("Pre run: \n{}".format(var))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    post_var = sess.run(var)
sess.close()
print("Post run: \n{}".format(post_var))

# %%
x_data = np.random.randn(5, 10)
w_data = np.random.randn(10, 1)

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=(None, 10))
    w = tf.placeholder(tf.float32, shape=(None, 1))
    b = tf.fill((5, 1), -1.0)
    xw = tf.matmul(x, w)

    xwb = xw + b
    s = tf.reduce_max(xwb)
    with tf.Session() as sess:
        outs = sess.run(s, feed_dict={x: x_data, w: w_data})

print("outs= {}".format(outs))

# %%

import numpy as np

# --- Create data and simulate results ---
x_data = np.random.rand(2000, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2

noise = np.random.randn(1, 2000) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise

NUM_STEPS = 100
wb_ = []
with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='weights')
        b = tf.Variable(0, dtype=tf.float32, name='bias')
        y_pred = tf.matmul(w, tf.transpose(x)) + b
    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    # Before starting, initialize the variables. We will 'run' this first.
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train, {x: x_data, y_true: y_data})
            if (step % 5 == 0):
                print(step, sess.run([w, b]))
                wb_.append(sess.run([w, b]))
        print(10, sess.run([w, b]))

# %%
N = 2000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Create data and simulate results ---

x_data = np.random.randn(N, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2
wxb = np.matmul(w_real, x_data.T) + b_real

y_data_pre_noise = sigmoid(wxb)
y_data = np.random.binomial(1, y_data_pre_noise)



NUM_STEPS = 100
with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='weights')
        b = tf.Variable(0, dtype=tf.float32, name='bias')
        y_pred = tf.matmul(w, tf.transpose(x)) + b
        y_pred = tf.sigmoid(y_pred)
        # y_pred = np.random.binomial(1, y_pred)
    with tf.name_scope('loss') as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train, {x: x_data, y_true: y_data})
            if(step%5 == 0):
                print(step, sess.run([w,b]))


