import os
import collections
import pickle
import numpy as np

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with open(input_file, "r") as f:
            data = f.read()  # Reads entire file
        counter = collections.Counter(data)  # dict of character counts

        # count_pairs is a list of [char, count] pairs sorted by count
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])

        # Transpose list of tuples into tuple of lists. Only keep sorted
        # char list
        self.chars, _ = list(zip(*count_pairs))  # char list sorted by count
        self.vocab_size = len(self.chars)

        # Make dict with key=char and value=index (0, len)
        # with 0 being with the most frequent char
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'w') as f:
            pickle.dump(self.chars, f)  # Write vocab

        # Get index (0, len) sorted with 0 being most freq char
        # for each char in index.
        # Put in array
        self.tensor = np.array(map(self.vocab.get, data))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file) as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)

    def create_batches(self):
        self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

