import logging
import os

import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils

from utils import file_opener, get_personality_files

logging.basicConfig(level=logging.INFO)

chars = []
raw_text = ""
training_data = get_personality_files()
save_path = ".{sep}model_artifacts{sep}".format(sep=os.sep)

# read training data
for file in file_opener(training_data):
    raw_text += file.read()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
logging.info("Total Characters: {}".format(n_chars))
logging.info("Total Vocab: {}".format(n_vocab))

# Prepare Training Data.
seq_length = 100
X = []
y = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i: i + seq_length]
    seq_out = raw_text[i + seq_length]
    X.append([char_to_int[char] for char in seq_in])
    y.append(char_to_int[seq_out])
n_patterns = len(X)
logging.info("Total Patterns: {}".format(n_patterns))

# reshape X to be [samples, time steps, features]
logging.info("Vectorizing Training Data...")
X = numpy.reshape(X, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(y)

# define the LSTM model
logging.info("Building Model...")
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
logging.info("Defining Checkpoint...")
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    os.path.join(save_path, filepath),
    monitor='loss', verbose=1,
    save_best_only=True, mode='min'
)
callbacks_list = [checkpoint]

# Train Model
logging.info("Training Model...")
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
