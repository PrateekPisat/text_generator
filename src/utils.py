import os
import math


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
        with open(file) as f:
            yield f


def get_training_files():
    training = list()
    dir_name = ".{sep}{dir}{sep}".format(sep=os.sep, dir="train")
    for _, __, files in os.walk(dir_name):
        for file in files:
            training += [dir_name + file]
    return training


def get_test_files():
    test = list()
    dir_name = ".{sep}{dir}{sep}".format(sep=os.sep, dir="test")
    for _, __, files in os.walk(dir_name):
        for file in files:
            test += [dir_name + file]
    return test
