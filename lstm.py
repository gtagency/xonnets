import tensorflow as tf
import numpy as np

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
from PRES_data import speeches

num_epochs = 100 # number of times to go through entire training set.
batch_size = len(speeches.source) # number of examples in each batch.
seq_len = min(len(src) for src in speeches.source) # length of each example sequence.
lstm_size = 1 # num units in LSTM = size of input at each timestep.

source_ = np.array(speeches.source)
source = [[[float(elem)] for elem in example] for example in source_]
expected = [src[1:] + [src[0]] for src in source]

# print [len(x) for x in source]

# X = source
# Y = expected
X = []
Y = []
for i in range(seq_len):
    X.append([])
    Y.append([])
    for j in range(batch_size):
        X[i].append(source[j][i])
        Y[i].append(expected[j][i])

lstm_cell = rnn_cell.BasicLSTMCell(lstm_size)

inputs = [tf.placeholder(tf.float32, [batch_size, lstm_size]) for _ in range(seq_len)]
results = [tf.placeholder(tf.float32, [batch_size, lstm_size]) for _ in range(seq_len)]
print(len(inputs))
print(len(results))

outputs, _ = rnn.rnn(lstm_cell, inputs, dtype=tf.float32)

cost = 0.0
for i in range(len(outputs)):
    cost += tf.reduce_mean(tf.pow(outputs[i]-results[i], 2))

train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for k in range(num_epochs):
        dict_in = {inputs[i]: np.array(X[i]) for i in range(len(inputs))}
        dict_in.update({results[i]: np.array(Y[i]) for i in range(seq_len)})

        print(list(dict_in.keys())[0])
        print(dict_in[list(dict_in.keys())[0]])

        sess.run(train_op, feed_dict=dict_in)
        cost_val = sess.run(cost)
        print('epoch', k, 'cost:', cost_val)
        print(sess.run(inputs[3]))
        print(sess.run(outputs[3]))
