# import os
from collections import defaultdict

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from utils import file_opener


def unigram_count(training_data, personality_files, gamma):
    """Return a dict of frequnency counts for unigrams."""
    count_dict = defaultdict(lambda: 0)
    # General English Model
    for comment in training_data:
        for sent in sent_tokenize(comment):
            for word in word_tokenize(sent):
                count_dict[word] += 1
    # Peronality Setup
    for file in file_opener(personality_files):
        for sent in sent_tokenize(file.read()):
            for word in word_tokenize(sent):
                count_dict[word] += gamma
    return count_dict


def bigram_count(training_data, personality_files, gamma):
    """Return a dict of frequency counts for all unique bigrams."""
    # Init a 2D dict. This is essentially a word-word matrix
    count_dict = defaultdict(lambda: defaultdict(lambda: 0))
    # General English model
    for comment in training_data:
        sents = sent_tokenize(comment)
        for sent in sents:
            words = word_tokenize(sent)
            # Get Bigrams
            bigrams = nltk.bigrams(
                words, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"
            )
            # Count frequency.
            for wi_1, wi in bigrams:
                count_dict[wi_1][wi] += 1
    # Peronality Setup
    for file in file_opener(personality_files):
        sents = sent_tokenize(file.read())
        for sent in sents:
            words = word_tokenize(sent)
            # Get Bigrams
            bigrams = nltk.bigrams(
                words, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"
            )
            # Count frequency.
            for wi_1, wi in bigrams:
                count_dict[wi_1][wi] += gamma
    # return results.
    return count_dict


def trigram_count(training_data, personality_files, gamma):
    """Return a dict of frequency counts for all unique trigrams."""
    # Init a 2D dict. This is essentially a bigram-word matrix
    count_dict = defaultdict(lambda: defaultdict(lambda: 0))
    # General English Model
    for comment in training_data:
        sents = sent_tokenize(comment)
        for sent in sents:
            words = word_tokenize(sent)
            trigrams = nltk.trigrams(
                words, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"
            )
            for wi_2, wi_1, wi in trigrams:
                count_dict[(wi_2, wi_1)][wi] += 1
    # Personality Model
    for file in file_opener(personality_files):
        sents = sent_tokenize(file.read())
        for sent in sents:
            words = word_tokenize(sent)
            trigrams = nltk.trigrams(
                words, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"
            )
            for wi_2, wi_1, wi in trigrams:
                count_dict[(wi_2, wi_1)][wi] += 1
    return count_dict


def adjust_counts(bigram_counts, trigram_counts, context, gamma):
    """Update the word frequencies based on context."""
    sents = sent_tokenize(context)
    unigram_counts = defaultdict(lambda: 0)
    for sent in sents:
        words = word_tokenize(sent)
        bigrams = nltk.bigrams(
            words, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"
        )
        trigrams = nltk.trigrams(
            words, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"
        )
        # Update Unigram Frequency
        for word in word_tokenize(sent):
            unigram_counts[word] += gamma
        # Update Bigram frequency.
        for wi_1, wi in bigrams:
            bigram_counts[wi_1][wi] += gamma
        # Update Trigram Frequency
        for wi_2, wi_1, wi in trigrams:
            trigram_counts[(wi_2, wi_1)][wi] += gamma
        # Return results
    return set(unigram_counts.keys()), bigram_counts, trigram_counts
