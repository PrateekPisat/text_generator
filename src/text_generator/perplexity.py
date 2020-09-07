import logging

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

from text_generator.utils import get_smallest_trigram_prob


def get_perplexity(model, file):
    """Calculate perplexity for `add_alpha_smoothing_model`.

    :param model: An instance of the triagram model.
    :param file: the file for which we calculate the perplexity.
    """
    with open(file) as f:
        buffer = f.read()
        total_words = len(word_tokenize(buffer))
        sents = sent_tokenize(buffer)
        perplexity = get_perplexity_for_file(sents, model, total_words)
        logging.warning("PP for file {} = {}".format(f.name, perplexity))
    return perplexity


def get_perplexity_for_file(sents, model, total_words):
    log_prob = 0
    smallest_prob = get_smallest_trigram_prob(model)

    for sent in sents:
        words = word_tokenize(sent)
        trigrams = nltk.trigrams(words)
        for wi_2, wi_1, wi in trigrams:
            context = tuple([wi_2, wi_1])
            prob = model[context][wi] if wi in model[context] else smallest_prob
            log_prob += np.log2(prob)

    perplexity = np.power(2, -(log_prob / total_words))
    return perplexity
