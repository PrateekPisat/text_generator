import json
import os

from keras.utils import np_utils


def get_smallest_trigram_prob(trigram_model):
    smallest_prob = 1
    for wi_2, wi_1 in trigram_model:
        for wi in trigram_model[tuple([wi_2, wi_1])]:
            prob = trigram_model[tuple([wi_2, wi_1])][wi]
            if prob < smallest_prob:
                smallest_prob = prob
    return smallest_prob


def get_smallest_birgram_prob(bigram_model, unigram_model, V):
    smallest_prob = 1
    for wi_1 in bigram_model:
        for wi in bigram_model[wi_1]:
            prob = (bigram_model[wi_1][wi] + 1) / (unigram_model[wi_1] + V)
            if prob < smallest_prob:
                smallest_prob = prob
    return smallest_prob


def get_smallest_unigram_prob(model, V):
    smallest_prob = 1
    total = sum(model.values())
    for w1 in model:
        prob = ((model[w1] + 1) / (total + V))
        if prob < smallest_prob:
            smallest_prob = prob
    return smallest_prob


def file_opener(files):
    for file in files:
        with open(file, encoding='utf-8') as f:
            yield f


def get_personality_files(name):
    training = list()
    dir_name = ".{sep}train{sep}{name}{sep}".format(sep=os.sep, name=name)
    for _, __, files in os.walk(dir_name):
        for file in files:
            training += [dir_name + file]
    return training


def get_training_data():
    f_path = ".{sep}train{sep}english{sep}RC_2007-02".format(sep=os.sep)
    with open(f_path) as f:
        buff = f.read()
        objs = buff.split("\n")
        objs = objs[:-1]
        json_objs = [json.loads(obj) for obj in objs]
        comments = [j_obj["body"] for j_obj in json_objs]
    return comments


def get_test_files():
    test = list()
    dir_name = ".{sep}{dir}{sep}".format(sep=os.sep, dir="test")
    for _, __, files in os.walk(dir_name):
        for file in files:
            test += [dir_name + file]
    return test


def create_word_list(words):
    word_list = []
    for word in words:
        word_list += [word.lower()]
    return word_list


def training_on_seed(model, seed, seq_length, char_to_int):
    n_chars = len(seed)
    # Prepare Training Data.
    dataX = []
    datay = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = seed[i: i + seq_length]
        seq_out = seed[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        datay.append(char_to_int[seq_out])
    X = np_utils.to_categorical(dataX)
    y = np_utils.to_categorical(datay)
    model.fit(X, y)
    return model
