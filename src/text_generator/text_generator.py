import math
import sys

import numpy

from text_generator.LSTM import train_lstm_model
from text_generator.utils import get_personality_files, get_training_data, training_on_seed


def gen_sentences(name, n_chars, seed, path=None):
    """Return a string of generated characters.

    :param name: name of direcotry that consists of the text corpus.
    :param n_chars: number of characters to generate.
    :param path: path to a pretrained model.
    """
    model, char_to_int, int_to_char, n_vocab, seq_length = train_lstm_model(name, False, path)
    dataX = []
    print("Seed Sentence Length Should Be >= {}".format(seq_length))
    print("Length of sentence you provided = {}".format(len(seed)))
    assert len(seed) >= seq_length
    # save weights
    model.save(".{sep}model_artifact{sep}temp_model")
    if len(seed) > seq_length:
        model = training_on_seed(model, seed, seq_length, char_to_int)
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


def get_perplexity(name, text, model_path):
    """Return perplexity for given text with respect to a given model.

    :param name: name of direcotry that consists of the text corpus.
    :param text: the text for which we generate perplexity.
    :param path: path to a pretrained model.
    """
    model, char_to_int, _, n_vocab, seq_length = train_lstm_model(
        name=name, train=False, path=model_path,
    )
    dataX = []
    seq_length = 100

    # Clean text
    clean_text = ""
    for c in text:
        if c in char_to_int:
            clean_text += c
    dataX.append([char_to_int[char] for char in clean_text])
    pattern = numpy.zeros((seq_length, n_vocab))

    for index, char in enumerate(clean_text[0:100]):
        pattern[index, char_to_int[char]] = 1
    total_log_probability = 0
    # generate characters
    for i in range(100, len(clean_text)):
        X = numpy.reshape(pattern, (1, seq_length, n_vocab))
        prediction = model.predict(X, verbose=0)
        index = char_to_int[clean_text[i]]
        result = prediction[0][index]
        total_log_probability += math.log(result)
        next_word = numpy.zeros(n_vocab)
        next_word[index] = 1
        pattern = numpy.append(pattern[1:], numpy.array([next_word]), axis=0)
    normalized_probability = math.exp(-1 * total_log_probability / len(clean_text))
    return normalized_probability


if __name__ == "__main__":
    args = sys.argv
    if args[1] == "-b" or args[1] == "--build_model":
        corpus_dir_name = args[2]
        epochs = int(args[3])
        model, _, __, ___, ____ = train_lstm_model(corpus_dir_name, train=True, epochs=epochs)
    elif args[1] == "-g" or args[1] == "--generate_sent":
        name = args[2]
        n_chars = int(args[3])
        seed = args[4]
        path = args[5]
        print(gen_sentences(name, n_chars=n_chars, seed=seed, path=path))
    elif args[1] == "-s" or args[1] == "--similarity":
        name = args[2]
        text = args[3]
        text = text.replace(r"\ufeff", "")
        text = text.lower()
        model_path = args[4]
        print(get_perplexity(name, text, model_path))
    elif args[1] == "--trigram-generate":
        name = args[2]
        n_lines = int(args[3])
        training_files = get_training_data()
        personality_data = get_personality_files(name)
        model = TrigramModel(training_files, personality_data, alpha=1)
        model.write_lines(n_lines)
    elif args[1] == "--trigram-similarity":
        name = args[2]
        file = args[3]
        training_files = get_training_data()
        personality_data = get_personality_files(name)
        model = TrigramModel(training_files, personality_data, alpha=1)
        print(model.get_simillarity(file))
