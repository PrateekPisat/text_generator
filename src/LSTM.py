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
raw_text = []
training_data = get_personality_files()
save_path = ".{sep}model_artifacts{sep}".format(sep=os.sep)

# read training data
for file in file_opener(training_data):
    words = file.read()
    words = list(
        filter(
            lambda x: x not in set(["\n", "\n\n", '\u2009', '\xa0', '\r\n']),
            words
        )
    )
    raw_text += words

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
logging.info("Total Characters: {}".format(n_chars))
logging.info("Total Vocab: {}".format(n_vocab))

# Prepare Training Data.
seq_length = 100
dataX = []
datay = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i: i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    datay.append(char_to_int[seq_out])
n_patterns = len(dataX)
logging.info("Total Patterns: {}".format(n_patterns))

# reshape X to be [samples, time steps, features]
# logging.info("Vectorizing Training Data...")
# X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# # normalize
# X = X / float(n_vocab)
X = np_utils.to_categorical(dataX)
# one hot encode the output variable
y = np_utils.to_categorical(datay)

# # define the LSTM model
logging.info("Building Model...")
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# # define the checkpoint
# logging.info("Defining Checkpoint...")
# filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(
#     os.path.join(save_path, filepath),
#     monitor='loss', verbose=1,
#     save_best_only=True, mode='min'
# )
# callbacks_list = [checkpoint]

# # Train Model
# logging.info("Training Model...")
# model.fit(
#     X,
#     y,
#     epochs=100,
#     batch_size=128,
#     callbacks=callbacks_list,
#     shuffle=True,
#     validation_split=0.2,
# )

# Generate Sentences
# load the network weights
filename = ".{sep}model_artifacts{sep}weights-improvement-100-0.2719.hdf5".format(sep=os.sep)
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)


# pick a random seed
start = numpy.random.randint(0, len(X) - 1)
pattern = X[start]
gen = ""
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in dataX[start]]), "\"")
# generate characters
for i in range(1000):
    # x = numpy.reshape(pattern, (1, len(pattern), 1))
    # x = x / float(n_vocab)
    x = numpy.reshape(pattern, (1, seq_length, n_vocab))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    gen += result
    next_word = numpy.zeros(n_vocab)
    next_word[index] = 1
    pattern = numpy.append(pattern[1:], numpy.array([next_word]), axis=0)
print("\n{}".format(gen))
print("\nDone.")
