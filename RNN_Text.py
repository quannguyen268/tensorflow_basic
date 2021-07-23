import numpy as np
import tensorflow as tf

tf = tf.compat.v1
tf.disable_v2_behavior()
batch_size = 128;
embedding_dimension = 64;
num_classes = 2
hidden_layer_size = 32;
time_steps = 6;
element_size = 1

digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                     6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

digit_to_word_map[0] = "PAD"

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

# %%

labels = [1] * 10000 + [0] * 10000

for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0] * 2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding

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


# %%
def get_sentence_batch(batch_size, data_x, data_y, data_seqlens):
    instance_indice = list(range(len(data_x)))
    np.random.shuffle(instance_indice)
    batch = instance_indice[:batch_size]
    x = [[word2index_map[word] for word in data_x[i].lower().split()]
         for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x, y, seqlens


inputs_ = tf.placeholder(tf.int32, shape=[batch_size, time_steps])
labels_ = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
seqlens_ = tf.placeholder(tf.float32, shape=[batch_size])

with tf.name_scope('embeddings'):
    embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dimension],
                                              -1.0, 1.0, ), name='embedding')
    embed = tf.nn.embedding_lookup(embedding, inputs_)

print(embed[:10])

with tf.name_scope('lstm'):
    lstm_cell = [tf.keras.layers.LSTMCell(hidden_layer_size) for _ in range (2)]
    cells = tf.keras.layers.StackedRNNCells(lstm_cell)
    outputs, states = tf.nn.dynamic_rnn(cells, embed, sequence_length=seqlens_,
                                        dtype=tf.float32)
weights = {
    'linear_layer': tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],
                                                    mean=0, stddev=.01))
}
bias = {
    'linear_layer': tf.Variable(tf.truncated_normal([num_classes], mean=0,
                                                    stddev=.01))
}

# Extract the last relevant output and use in a linear layer
final_output = tf.matmul(states[1][1], weights['linear_layer'] + bias['linear_layer'])
softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=labels_)
loss_crossentropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss_crossentropy)

correct_prediction = tf.equal(tf.argmax(labels_, 1), tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size, train_x, train_y, train_seqlens)
        sess.run(train_step, feed_dict={inputs_: x_batch, labels_: y_batch, seqlens_: seqlen_batch})
        print(seqlen_batch)
        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={inputs_: x_batch, labels_: y_batch, seqlens_: seqlen_batch})

            print("Accuracy at %d: %.5f" % (step, acc))

    for test_batch in range(5):
        x_test, y_test, seqlen_test = get_sentence_batch(batch_size, test_x, test_y, test_seqlens)
        batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
                                         feed_dict={inputs_: x_test, labels_: y_test,
                                                    seqlens_: seqlen_test})
        print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))

    output_example = sess.run([outputs], feed_dict={inputs_: x_test, labels_: y_test,
                                                  seqlens_: seqlen_test})
    states_example = sess.run([states[1]], feed_dict={inputs_: x_test, labels_: y_test,
                                                    seqlens_: seqlen_test})

# %%
print(seqlen_test[1])
print(output_example[0][1])
