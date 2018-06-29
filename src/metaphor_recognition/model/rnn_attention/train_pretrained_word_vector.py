#!/usr/bin/python
"""
Toy example of attention layer use

Train RNN (GRU) on IMDB dataset (binary classification)
Learning and hyper-parameters were not tuned; script serves as an example 
"""
from __future__ import print_function, division

import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tqdm import tqdm
from keras.utils import np_utils
from keras.preprocessing import text, sequence

import src.metaphor_recognition.model.rnn_attention.attention as attention
from src.metaphor_recognition.model.rnn_attention.utils import get_vocabulary_size, fit_in_vocabulary, zero_pad, batch_generator
import src.original_data.data_path as data_path
import src.common.file_utils as file_utils

NUM_WORDS = 10000
INDEX_FROM = 3
SEQUENCE_LENGTH = 90
EMBEDDING_DIM = 300
HIDDEN_SIZE = 32
ATTENTION_SIZE = 32
KEEP_PROB = 0.5
BATCH_SIZE = 64
NUM_EPOCHS = 12  # Model easily overfits without pre-trained words embeddings, that's why train for a few epochs
DELTA = 0.5
MODEL_PATH = './model/model'

EMBEDDING_FILE = r'D:\公司\中文隐喻识别与情感分析\data\sgns.baidubaike.bigram-char\sgns.baidubaike.bigram-char.correct'

# Load the data set
# C:\Users\liyuncong\.keras\datasets
# (X_train, y_train), (X_test, y_test) = imdb.load_data(path='imdb.npz', num_words=NUM_WORDS, index_from=INDEX_FROM)


def get_label_and_text(line):
    split_index = line.index(' ')
    label = line[:split_index]
    text = line[split_index + 1:]
    return (label, text)


train_lines = file_utils.read_all_lines(data_path.metaphor_data_base_dir + 'metaphor_recognition.nn.train')
validation_lines = file_utils.read_all_lines(data_path.metaphor_data_base_dir + 'metaphor_recognition.nn.validation')


X_train = [get_label_and_text(line)[1] for line in train_lines if line]
y_train = [int(get_label_and_text(line)[0]) for line in train_lines if line]
X_test = [get_label_and_text(line)[1] for line in validation_lines if line]
y_test = [int(get_label_and_text(line)[0]) for line in validation_lines if line]

nb_classes = 3
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

max_features = 10000
maxlen = 90
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train + X_test)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Sequences pre-processing
vocabulary_size = get_vocabulary_size(X_train)
X_test = fit_in_vocabulary(X_test, vocabulary_size)
X_train = zero_pad(X_train, SEQUENCE_LENGTH)
X_test = zero_pad(X_test, SEQUENCE_LENGTH)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


word_index = tokenizer.word_index

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding='utf-8') if o.strip().split()[0] in word_index)

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

nb_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBEDDING_DIM))
embedding_matrix = embedding_matrix.astype(dtype='float32')
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Different placeholders
with tf.name_scope('Inputs'):
    batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
    target_ph = tf.placeholder(tf.float32, [None, 3], name='target_ph')
    seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

# Embedding layer
with tf.name_scope('Embedding_layer'):
    embeddings_var = tf.Variable(embedding_matrix, trainable=False)
    tf.summary.histogram('embeddings_var', embeddings_var)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

# (Bi-)RNN layer(-s)
rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                        inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
tf.summary.histogram('RNN_outputs', rnn_outputs)

# Attention layer
with tf.name_scope('Attention_layer'):
    attention_output, alphas = attention.attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
    tf.summary.histogram('alphas', alphas)

# Dropout
drop = tf.nn.dropout(attention_output, keep_prob_ph)

# Fully connected layer
with tf.name_scope('Fully_connected_layer'):
    W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2, nb_classes], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
    b = tf.Variable(tf.constant(0., shape=[nb_classes]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    y_hat = tf.squeeze(y_hat)
    tf.summary.histogram('W', W)

with tf.name_scope('Metrics'):
    # Cross-entropy loss and optimizer initialization
    # tf.nn.softmax_cross_entropy_with_logits() for muliti_class
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_ph, logits=y_hat))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    # Accuracy metric
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), target_ph), tf.float32))

    correct_prediction = tf.equal(tf.argmax(target_ph, 1), tf.argmax(y_hat, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

# Batch generators
train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE)
test_batch_generator = batch_generator(X_test, y_test, BATCH_SIZE)

train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
test_writer = tf.summary.FileWriter('./logdir/test', accuracy.graph)

session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

saver = tf.train.Saver()

if __name__ == "__main__":
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        for epoch in range(NUM_EPOCHS):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = X_train.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(train_batch_generator)
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                loss_tr, acc, _, summary = sess.run([loss, accuracy, optimizer, merged],
                                                    feed_dict={batch_ph: x_batch,
                                                               target_ph: y_batch,
                                                               seq_len_ph: seq_len,
                                                               keep_prob_ph: KEEP_PROB})
                accuracy_train += acc
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                train_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_train /= num_batches

            # Testing
            num_batches = X_test.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(test_batch_generator)
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                loss_test_batch, acc, summary = sess.run([loss, accuracy, merged],
                                                         feed_dict={batch_ph: x_batch,
                                                                    target_ph: y_batch,
                                                                    seq_len_ph: seq_len,
                                                                    keep_prob_ph: 1.0})
                accuracy_test += acc
                loss_test += loss_test_batch
                test_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_test /= num_batches
            loss_test /= num_batches

            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                loss_train, loss_test, accuracy_train, accuracy_test
            ))
        train_writer.close()
        test_writer.close()
        saver.save(sess, MODEL_PATH)
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
