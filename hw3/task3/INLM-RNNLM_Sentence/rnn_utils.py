import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import json
import glob

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, min_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding
        self.min_length = min_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            #self.char_file("./data/words_dictionary.json", input_file, self.seq_length, self.min_length)
            self.sent_file1("./data/texts", input_file, self.seq_length, self.min_length)
            self.preprocess(input_file, vocab_file, tensor_file)

        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()
    

    # Cleans non-ascii, nothing else
    def sent_file1(self, path, input_f, seq_length, min_length):
        
        # collect texts from path
        files = glob.glob( os.path.join( path, "*.txt" ) )
        texts = ""
        for file in files:
            with open(file, 'r') as f:
                texts = texts + f.read()
        print(str(len(files)) + " files collected, char count: "+str(len(texts)))
    
        # clean text, may add more in the future
        texts = texts.decode("ascii","ignore").encode()

        # write to file
        with open(input_f, 'w') as f:
            print("Writing to input.txt")
            f.write(texts)
    
    # Pad sentence, don't pad word
    def sent_file2(self, path, input_f, seq_length, min_length):
        
        # collect texts from path
        files = glob.glob( os.path.join( path, "*.txt" ) )
        texts = ""
        for file in files:
            with open(file, 'r') as f:
                texts = texts + f.read()
        print(str(len(files)) + " files collected, char count: "+str(len(texts)))
        
        # clean text, one sent in each seq
        texts = texts.decode("ascii","ignore").encode()
        terminators = ". ; : ! ? \n".split(" ")

        # pad into seq
        cut = 0
        t=0
        lines=[]
        for t in range(len(texts)):
            if texts[t] in terminators:
                sent = texts[cut:t]+texts[t]
                t = t+1
                cut = t
                if len(sent) > min_length and len(sent) < seq_length:
                    sent = sent + ("`")*(seq_length - len(sent))
                    lines.append(sent)
                    if len(lines)%50 == 0:
                        print("Generating data: " + str(len(lines)))


        # write to file
        with open(input_f, 'w') as f:
            print("Writing to input.txt")
            for line in lines:
                print(line)
                f.write(line)
    

    def char_file(self,path, input_f, seq_length, min_length):
        lines = []
        for i in range(3):
            with open(path, 'r') as f:
                valid_words = json.load(f)
                e_w_list = list(valid_words.keys())
                for line in e_w_list:
                    line = tuple(line.encode("ascii"))
                    if len(line) <= seq_length and len(line) > min_length:
                        lines.append(line + (("`",) * (seq_length - len(line))))
                        if len(lines) % 5000 == 0:
                            print("Generating data: " + str(len(lines)))

                    if len(lines) == 1000000:
                        break
        with open(input_f, 'w') as f:
            print("Writing to input.txt")
            for line in lines:
                line = "".join(line)
                print(line)
                f.write(line)

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
