import logging
import os

from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from utils import build_lstm_model, file_opener, get_personality_files

logging.basicConfig(level=logging.INFO)

chars = []
raw_text = []
training_data = get_personality_files()
save_path = ".{sep}model_artifacts{sep}".format(sep=os.sep)

# read training data
for file in file_opener(training_data):
    words = file.read()
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

logging.info("Vectorizing Training Data...")
# one hot encode the output variable
X = np_utils.to_categorical(dataX)
y = np_utils.to_categorical(datay)

# define the LSTM model
logging.info("Building Model...")
model = build_lstm_model(seq_length, n_vocab)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
logging.info("Defining Checkpoint...")
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    os.path.join(save_path, filepath),
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    period=25,
)
callbacks_list = [checkpoint]

# Train Model
logging.info("Training Model...")
model.fit(
    X,
    y,
    epochs=75,
    batch_size=128,
    callbacks=callbacks_list,
    shuffle=True,
    validation_split=0.2,
)
