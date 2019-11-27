import os

import numpy

from models import TrigramModel
from utils import (
    build_lstm_model,
    file_opener,
    get_personality_files,
    get_test_files,
    get_training_data,
    sample,
)

seq_length = 100
chars = []
raw_text = []
training_data = get_personality_files()
save_path = ".{sep}model_artifacts{sep}".format(sep=os.sep)

# read training data
for file in file_opener(training_data):
    words = file.read()
    # words = words.replace("‘", "'").replace('“')
    raw_text += words

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

# filename = ".{sep}models{sep}trigram_model".format(sep=os.sep)
# charles = get_personality_files()
# training_data = get_training_data()
# test_files = get_test_files()
# model = TrigramModel(training_data=training_data, personality=charles)
# for file in test_files:
#     print("SimScore = {}".format(model.get_simillarity(file)))
# model.write_lines(10)

# Generate Sentences
# load the network weights

filename = ".{sep}model_artifacts{sep}weights-improvement-75-0.4099.hdf5".format(sep=os.sep)
print("Working with model : weights-improvement-75-0.4099.hdf5")
model = build_lstm_model(seq_length, n_vocab)
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# pick a random seed
dataX = []
start = numpy.random.randint(0, (n_chars - seq_length - 1))
seed = 'I started walking towards the bar and saw that my car was missing so I called the police and told th'
assert len(seed) == seq_length
print("Seed: {}".format(seed))
dataX.append([char_to_int[char] for char in seed])
pattern = numpy.zeros((seq_length, n_vocab), dtype=numpy.bool)
for index, char in enumerate(seed):
    pattern[index, char_to_int[char]] = 1
gen = ""
# generate characters
for i in range(500):
    X = numpy.reshape(pattern, (1, seq_length, n_vocab))
    prediction = model.predict(X, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    gen += result
    next_word = numpy.zeros(n_vocab)
    next_word[index] = 1
    pattern = numpy.append(pattern[1:], numpy.array([next_word]), axis=0)
print("\nDone.")
print("{}{}".format(seed, gen))