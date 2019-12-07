import os
import sys

import numpy
from nltk.tokenize import word_tokenize

from LSTM import train_lstm_model
from summary import get_summary

import math

# seed = "as a subject for the remarks of the evening, the perpetuation of our  political institutions, is sel"
# seed = "and the shoemaker was not allowed by us to be a husbandman, or a weaver, or a builder--in order that"


def gen_sentences(n_chars, seed, path=None):
    model, char_to_int, int_to_char, n_vocab, seq_length = train_lstm_model(
        "charles",
        train=False,
        path=".{sep}model_artifacts{sep}char-gen-model-200-0.2899.hdf5".format(sep=os.sep),
    )
    dataX = []
    seq_length = 100
    print("Seed Sentence Length Should Be = {}".format(seq_length))
    print(len(seed))
    assert len(seed) == seq_length
    dataX.append([char_to_int[char] for char in seed])
    pattern = numpy.zeros((seq_length, n_vocab))

    for index, char in enumerate(seed):
        pattern[index, char_to_int[char]] = 1
    gen = ""
    # generate characters
    for _ in range(n_chars):
        X = numpy.reshape(pattern, (1, seq_length, n_vocab))
        prediction = model.predict(X, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        gen += result
        next_word = numpy.zeros(n_vocab)
        next_word[index] = 1
        pattern = numpy.append(pattern[1:], numpy.array([next_word]), axis=0)
    return "{}{}".format(seed, gen)

def get_perplexity(text):
    model, char_to_int, int_to_char, n_vocab, seq_length = train_lstm_model(
        "charles",
        train=False,
        path=".{sep}model_artifacts{sep}char-gen-model-200-0.2899.hdf5".format(sep=os.sep),
    )
    dataX = []
    seq_length = 100
    
    dataX.append([char_to_int[char] for char in text])
    pattern = numpy.zeros((seq_length, n_vocab))

    for index, char in enumerate(text[0:100]):
        pattern[index, char_to_int[char]] = 1
    total_log_probability = 0
    # generate characters
    for i in range(100, len(text)):
        X = numpy.reshape(pattern, (1, seq_length, n_vocab))
        prediction = model.predict(X, verbose=0)
        print(prediction)
        index = char_to_int[text[i]]
        result = prediction[0][index]
        print(index, result)
        total_log_probability += math.log(result)
        next_word = numpy.zeros(n_vocab)
        next_word[index] = 1
        pattern = numpy.append(pattern[1:], numpy.array([next_word]), axis=0)
    normalized_probability = math.exp(-1 * total_log_probability / len(text))
    print("total probability", normalized_probability)
    return normalized_probability


# Doc Similarity
def doc_sim(test_file):
    pass


if __name__ == "__main__":
    args = sys.argv
    if args[1] == "-g" or args[1] == "--generate_sent":
        n_chars = int(args[2])
        seed = args[3]
        print(gen_sentences(n_chars, seed))
    elif args[1] == "-s" or args[1] == "--similarity":
        text = args[2]
        text = text.replace(r'\ufeff', '')
        text = text.lower()
        get_perplexity(text)
