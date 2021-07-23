import zipfile

import numpy as np
import tensorflow  as tf
from tqdm import tqdm
print(tf.__version__)

tf = tf.compat.v1

tf.disable_v2_behavior()


PRE_TRAINED = True
GLOVE_SIZE = 300
batch_size = 128
embedding_dimension = 64
num_classes = 2
hidden_layer_size = 32
times_steps = 6
path_to_glove = '/home/quan/PycharmProjects/tensorflow_basic/glove/glove.840B.300d.txt'


digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                     6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

digit_to_word_map[0] = "PAD_TOKEN"

even_sentences = []
odd_sentences = []
seqlens = []

for i in range(10000):
    rand_seq_len = np.random.choice(range(3, 7))
    seqlens.append(rand_seq_len)
    rand_odd_ints = np.random.choice(range(1, 10, 2), rand_seq_len)
    rand_even_ints = np.random.choice(range(2, 10, 2), rand_seq_len)

    # Padding
    if rand_seq_len < 6:
        rand_odd_ints = np.append(rand_odd_ints, [0] * (6 - rand_seq_len))
        rand_even_ints = np.append(rand_even_ints, [0] * (6 - rand_seq_len))

    even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
    odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))

data = even_sentences + odd_sentences
# Same seq lengths for even, odd sentences
seqlens *= 2

# Map from words to indices
word2index_map = {}
index = 0
for sent in data:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

# Inverse map
index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)



labels = [1] * 10000 + [0] * 10000

for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0] * 2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding




def get_glove(path_to_glove, word2index_map):
    embedding_weights = {}
    count_all_words = 0

    f = open(path_to_glove, 'r', encoding='utf-8')
    for line in tqdm(f):
        vals = line.split(' ')
        word = vals[0]
        if word in word2index_map:
            print(word)
            count_all_words += 1
            coefs = np.asarray([float(val) for val in vals[1:]])
            coefs /= np.linalg.norm(coefs)
            embedding_weights[word] = coefs
        if count_all_words == vocabulary_size - 1:
            break
    return embedding_weights
word2embedding_dict = get_glove(path_to_glove, word2index_map)

embedding_matrix = np.zeros((vocabulary_size ,GLOVE_SIZE))
for word,index in word2index_map.items():
    if not word == "pad_token":
        word_embedding = word2embedding_dict[word]
        embedding_matrix[index,:] = word_embedding

data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]
labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]
train_x = data[:10000]
train_y = labels[:10000]
train_seqlens = seqlens[:10000]
test_x = data[10000:]
test_y = labels[10000:]

test_seqlens = seqlens[10000:]
def get_sentence_batch(batch_size,data_x, data_y,data_seqlens):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i].split()] for i in batch]
    print(x)
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x,y,seqlens

#%%

# x = [word2index_map[str(word)] for word in train_x[0].split()]
# print(x)
print(word2index_map)
# x,y = get_sentence_batch(10, train_x, train_y, train_seqlens)

#%%

with tf.Graph().as_default():

    inputs_ = tf.placeholder(tf.int32, shape=[batch_size, times_steps])
    embedding_placeholder = tf.placeholder(tf.float32, shape=[vocabulary_size, GLOVE_SIZE])
    labels_ = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
    seqlens_ = tf.placeholder(tf.int32, shape=[batch_size])

    if PRE_TRAINED:
        embeddings = tf.Variable(tf.constant(0.0, shape=[vocabulary_size, GLOVE_SIZE]),
                                 trainable=True)

        # if using pretrained embeddings, assign them to the embeddings variable
        embedding_init = embeddings.assign(embedding_placeholder)
        embed = tf.nn.embedding_lookup(embeddings, inputs_)

    else:
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dimension],
                                                   -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, inputs_)

    with tf.name_scope('biGRU'):
        with tf.variable_scope('forward'):
            gru_fw_cell = tf.keras.layers.GRUCell(hidden_layer_size, dropout=0.1)

        with tf.variable_scope('backward'):
            gru_bw_cell = tf.keras.layers.GRUCell(hidden_layer_size, dropout=0.1)


        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell,
                                                          cell_bw=gru_bw_cell,
                                                          inputs=embed,
                                                          sequence_length=seqlens_,

                                                          dtype=tf.float32,scope='biGRU')

    states = tf.concat(values=states, axis=1)

    linear_layer = {
        'weights': tf.Variable(tf.truncated_normal([2*hidden_layer_size, num_classes],
                                                   mean=0, stddev=.01)),
        'biases': tf.Variable(tf.truncated_normal([2*hidden_layer_size, num_classes],
                                                   mean=0, stddev=.01))
    }

    # extract the final state and use in linear layer
    final_output = tf.matmul(states, linear_layer['weights'] + linear_layer['biases'])
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=labels_)
    cross_entropy = tf.reduce_mean(softmax)

    train_step = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(labels_, 1), tf.argmax(final_output, 1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(embedding_init, feed_dict={embedding_placeholder: embedding_matrix})

        for step in range(1000):
            x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size, train_x, train_y, train_seqlens)
            sess.run(train_step, feed_dict={inputs_: x_batch, labels_: y_batch, seqlens_: seqlen_batch})

            if step % 100 == 0:
                acc = sess.run(accuracy, feed_dict={inputs_: x_batch, labels_: y_batch, seqlens_: seqlen_batch})

                print("Accuracy at %d: %.5f" % (step, acc))

        for test_batch in range(5):
            x_test, y_test, seqlen_test = get_sentence_batch(batch_size, test_x, test_y, test_seqlens)
            batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
                                             feed_dict={inputs_: x_test, labels_: y_test,
                                                        seqlens_: seqlen_test})
            print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))






