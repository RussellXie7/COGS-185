
LSTMLM_Sentence Branch Usage (Updated: 4/23/2018)
=======================================================

Files 
<ul>
    <li>rnn_utils.py</li>
        <ul>
             <li>load and pre-process data </li>
             <li>sent_file1 for no padding</li>
             <li>sent_file2 for one sentence per sequence w/ padding</li>
             <li>input texts should be placed in data/texts with .txt suffix</li>
        </ul>
    <li>rnn_train.py</li>
        <ul>
             <li>Create and Train the RNN </li>
        </ul>
    <li>rnn_model.py</li>
         <ul>
             <li>RNN Model</li>
        </ul>
    <li>rnn_sample.py</li>
         <ul>
             <li>Sample generated sequences from checkpoints</li>
        </ul>
</ul>

Folders (create them if you don't have those in your project folder)
<ul>
    <li>data</li>
        <ul>
             <li>The folder stores the needed training data</li>
        </ul>
    <li>logs</li>
        <ul>
             <li>The folder stores the log information</li>
        </ul>
    <li>save</li>
        <ul>
             <li>The folder stores checkpoints you saved</li>
        </ul>
</ul>


Data Preprocessing:
=======================================================

For char-level sentence generation, put all input texts in data/texts directory with .txt suffix

Or you may define the dir in rnn_utils.py: line 24

You may also choose between functions sent_file1 and sent_file2

For sent_file1, all texts are read in as a whole chunk

For sent_file2, each sentence is padded with \` to ensure one sentence in each sequence

Every time you want to generate new input.txt for training, please remember to delete all the files in data_dir.

Side Note: the sentence chunking is rather primitive now. Adjust the terminater list for sent_file2 if you need. 


Training:
=======================================================

To run CHAR-RNN or CHAR-LSTM for training, use command "python rnn_train.py" with corresponding arguments.

To clean up saved models AND preprocessed inputs, type "make clean" in the terminal
[ Just a easy usage despite I seem to be the only person using a terminal here... ]

For more reference, please see https://github.com/sherjilozair/char-rnn-tensorflow

Input arguments you may want to play around with:
1. data_dir: data directory containing input.txt parsed using rnn_utils.py. 
2. model: rnn, gru, lstm, or nas (Only lstm works here)
3. batch_size: minibatch size
4. seq_length: length of each input data
5. min_length: min length of input data
6. num_epochs: number of training iterations
7. save_every: checkpoint saving frequency. The sampling process can be long for each checkpoint, so don't save too many checkpoints.

These arguments were changed for word generation. To do sentence generation, please refer to the original github repo for the original values.

DO NOT CHANGE THE STRUCTURE AND HYPER-PARAMETERS OF THE MODEL!


Sampling: 
=======================================================

To sample the generated sequence from checkpoints, use command "python rnn_sample.py".

Input arguments you may want to play around with:
1. n: number of characters you want to generate, seq_length - 1.
2. prime: head of the generated sequence.

Remember to change the line for different input:

(line 68) with open("./data/words_dictionary.json", 'r') as f:

Note: all generation outputs are found in the negative documents, because we do not have a evaluation method yet.


