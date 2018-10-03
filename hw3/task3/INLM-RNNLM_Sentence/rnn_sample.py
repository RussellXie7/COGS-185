from __future__ import print_function
import tensorflow as tf

import string
import random

import argparse
import os
from six.moves import cPickle

from rnn_model import Model

from six import text_type

import json
from tensorflow.python.platform import gfile

import sys

def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=50,
                        help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)

def split_pos_neg_words(samples, ground_truth):

    result_words = get_intermediate_data(samples)
    pos = []
    neg = []
    for i, item in enumerate(result_words):
        if item.rstrip() in ground_truth:
            pos.append(i)
        else:
            neg.append(i)
    pos_words = [samples[x] for x in pos]
    neg_words = [samples[x] for x in neg]
    return pos_words, neg_words, len(pos_words), len(neg_words)

def get_intermediate_data(samples):

    result = []
    for s in samples:
        s = "".join(s)
        s = s.replace("`", " ")
        result.append(s)
    return result

def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        with open("./data/words_dictionary.json", 'r') as f:
            valid_words = json.load(f)
        for model_path in ckpt.all_model_checkpoint_paths:
            print(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, model_path)
                samples = []
                sample_size = 200
                for i in range(sample_size):
                    prime = random.choice(string.letters).lower().decode('utf-8')
                    samp = model.sample(sess, chars, vocab, args.n, prime,
                                        args.sample).encode('utf-8')
                    samples.append(samp)
                    print("Generating sample: " + str(i))
                

                batch_new_pos, batch_new_neg, batch_pos_size, batch_neg_size = split_pos_neg_words(samples, valid_words)
                file_name = model_path.replace("/", "-")
                with gfile.GFile('./rnn_result_10/pos_samples-'+file_name+'.txt', mode="w") as f:
                    f.write("Pos size: " + str(batch_pos_size) + "\n")
                    f.write("Pos rate: " + str(float(batch_pos_size) / sample_size) + "\n")
                    for s in batch_new_pos:
                        s = "".join(s)
                        f.write(s + "\n")
                    print("pos sample created.")

                with gfile.GFile('./rnn_result_10/neg_samples-'+file_name+'.txt', mode="w") as f:
                    f.write("Neg size: " + str(batch_neg_size) + "\n")
                    for s in batch_new_neg:
                        s = "".join(s)
                        f.write(s + "\n")
                    print("neg sample created.")

if __name__ == '__main__':
    main()
