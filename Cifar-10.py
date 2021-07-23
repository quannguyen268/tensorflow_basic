import tensorflow as tf
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from keras.datasets import cifar10

class CifarLoader(object):
    def __init__(self, soure_files):
        self._source = soure_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d['data'] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).tranpose(0, 2, 3, 1)\
            .astype(float) / 255

    def nex_batch(self, batch_size):
        x,y = self.images[self._i: self._i + batch_size], self.labels[self._i: self._i + batch_size]

        self._i = (self._i + batch_size) % len(self.images)


DATA_PATH = ''

def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as f:
        dict = pickle.load(f)
        return dict

def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

class CifaDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i)
                                  for i in range(1, 6)]).load()

        self.test = CifarLoader(["test_batch"]).load()

def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
                    for i in range(size)])
    plt.imshow(im)
    plt.show()

(x_train, y_train),( x_test, y_test) = cifar10.load_data()

display_cifar(x_train, 10)
#%%

import tensorflow as tf
import numpy as np

a = np.array([[1,3,1],
              [4,5,6],
              [7,8,9]])

b = tf.argmax(a, 1)

sess = tf.InteractiveSession()
out = sess.run(b)
print(out)