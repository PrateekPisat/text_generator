import logging
import os
import numpy as np
import random
import sys
import time
from collections import Counter

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.models import Sequential, Model
from nltk.tokenize import word_tokenize
from six.moves import cPickle

from utils import create_word_list, get_personality_files


import spacy
nlp = spacy.load('en_core_web_sm')

# define parameters used in the tutorial
save_dir = '.{sep}model_artifacts{sep}'.format(sep=os.sep)
vocab_file = os.path.join(save_dir, "words_vocab.pkl")
sequences_step = 1
seq_length = 30

word_list = []
training_data = get_personality_files()

for file_name in training_data:
    # read data
    with open(file_name, "r") as f:
        data = f.read()
    # create sentences
    words = word_tokenize(data)
    wl = create_word_list(words)
    word_list = word_list + wl

# Build Dictionary
# count the number of words
word_counts = Counter(word_list)

# Mapping from index to word : that's the vocabulary
vocabulary_inv = list(sorted([x[0] for x in word_counts.most_common()]))

# Mapping from word to index
vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]

# size of the vocabulary
vocab_size = len(words)

logging.warning("Creating Sentences.")
# create sequences
sequences = []
next_words = []
for i in range(0, len(word_list) - seq_length, sequences_step):
    sequences.append(word_list[i: i + seq_length])
    next_words.append(word_list[i + seq_length])
logging.warning("Done Creating Sentences.")


# Vectorize the sequences and next_words
logging.warning("Vectorizing Sentences Sentences.")
X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence):
        X[i, t, vocab[word]] = 1
    y[i, vocab[next_words[i]]] = 1
logging.warning("Done Vectorizing Sentences Sentences.")


def bidirectional_lstm_model(seq_length, vocab_size):
    # Hyperparameters
    rnn_size = 256
    seq_length = 30
    learning_rate = 0.001
    print('Build LSTM model.')
    model = Sequential()
    model.add(
        Bidirectional(LSTM(rnn_size, activation="relu"), input_shape=(seq_length, vocab_size))
    )
    model.add(Dropout(0.6))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate)
    model.compile(
        loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy]
    )
    print("model built!")
    return model


# Build Model
logging.warning("Building Model.")
md = bidirectional_lstm_model(seq_length, vocab_size)
md.summary()
logging.warning("Done Building Model.")

# Train The model
logging.warning("Trainig Model.")
batch_size = 32
num_epochs = 50

callbacks = [
    EarlyStopping(patience=4, monitor='val_loss'),
    ModelCheckpoint(
        filepath=save_dir + "/" + 'my_model_gen_sentences.{epoch:02d}-{val_loss:.2f}.hdf5',
        monitor='val_loss',
        verbose=0,
        mode='auto',
        period=2
    )
]
# fit the model
history = md.fit(
    X, y,
    batch_size=batch_size,
    shuffle=True,
    epochs=num_epochs,
    callbacks=callbacks,
    validation_split=0.2
)

# save the model
md.save(save_dir + "/" + 'my_model_generate_sentences.h5')
logging.warning("Done Trainig Model.")
