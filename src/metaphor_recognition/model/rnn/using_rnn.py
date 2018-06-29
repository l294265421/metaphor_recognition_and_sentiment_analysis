# coding=utf-8
import numpy as np

np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.utils import np_utils

import src.original_data.data_path as data_path
import src.common.file_utils as file_utils

import warnings

warnings.filterwarnings('ignore')

import os

os.environ['OMP_NUM_THREADS'] = '4'

EMBEDDING_FILE = r'D:\公司\中文隐喻识别与情感分析\data\sgns.baidubaike.bigram-char\sgns.baidubaike.bigram-char.correct'


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
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


word_index = tokenizer.word_index

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding='utf-8') if o.strip().split()[0] in word_index)

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

nb_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))


def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(LSTM(40, return_sequences=True))(x)
    x = Bidirectional(GRU(40, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(3, activation="softmax")(conc)

    model = Model(inputs=inp, outputs=outp)
    adam = optimizers.adam(clipvalue=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model
model = get_model()

batch_size = 32
epochs = 15
# X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)

early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
ra_val = RocAucEvaluation(validation_data=(x_test, y_test), interval=1)
callbacks_list = [ra_val, early]
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks = callbacks_list,
          verbose=1)
# y_pred = model.predict(x_test,batch_size=1024,verbose=1)