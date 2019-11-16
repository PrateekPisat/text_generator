from collections import defaultdict

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from utils import file_opener


def unigram_count(files):
    """Return a dict of frequnency counts for unigrams."""
    count_dict = defaultdict(lambda: 0)

    for file in file_opener(files):
        for sent in sent_tokenize(file.read()):
            for word in word_tokenize(sent):
                count_dict[word] += 1
    return count_dict


def bigram_count(files):
    """Return a dict of frequency counts for all unique bigrams."""
    # Init a 2D dict. This is essentially a word-word matrix
    count_dict = defaultdict(lambda: defaultdict(lambda: 0))
    # Iterate over each file.
    for file in file_opener(files):
        sents = sent_tokenize(file.read())
        for sent in sents:
            words = word_tokenize(sent)
            # Get Bigrams
            bigrams = nltk.bigrams(
                words, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"
            )
            # Count frequency.
            for wi_1, wi in bigrams:
                count_dict[wi_1][wi] += 1
    # return results.
    return count_dict


def trigram_count(files):
    """Return a dict of frequency counts for all unique trigrams."""
    # Init a 2D dict. This is essentially a bigram-word matrix
    count_dict = defaultdict(lambda: defaultdict(lambda: 0))
    for file in file_opener(files):
        sents = sent_tokenize(file.read())
        for sent in sents:
            words = word_tokenize(sent)
            trigrams = nltk.trigrams(
                words, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"
            )
            for wi_2, wi_1, wi in trigrams:
                count_dict[(wi_2, wi_1)][wi] += 1
    return count_dict
