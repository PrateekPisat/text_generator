import random
from collections import defaultdict

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from nltk.tokenize import word_tokenize

from counts import adjust_counts, bigram_count, trigram_count, unigram_count
from perplexity import get_perplexity


class TrigramModel:
    """Represents a trigram model."""
    def __init__(self, training_data, personality, alpha=1, gamma=10):
        """ Return a trained trigram model.

        :param training_data: A list of files names which we use to train our model's personality.
        :param personality: A list of string which we use to train our model.
        :param alpha: alpha for add alpha smoothing.
        :param gamma: value to influence the personality n-gram counts.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.vocab = set(unigram_count(training_data, personality, gamma).keys())
        self.bigram_counts = bigram_count(training_data, personality, gamma)
        self.trigram_counts = trigram_count(training_data, personality, gamma)
        self.trigram_model = self.fit(self.trigram_counts, self.bigram_counts)

    def write_lines(self, n_lines, context=None):
        # init
        lines = []
        curr = tuple(["<s>", "<s>"])
        next_word = ""
        # Save current values:
        stack = [(self.vocab, self.bigram_counts, self.trigram_counts, self.trigram_model)]
        if context:
            c_words = word_tokenize(context)
            if len(c_words) >= 2:
                curr = tuple([c_words[-2], c_words[-1]])
            else:
                curr = tuple("<s>", c_words[-1])
            # Adjust probability
            self.vocab, self.bigram_counts, self.triram_couns = adjust_counts(
                self.bigram_counts, self.trigram_counts, context, self.gamma
            )
            self.trigram_model = self.fit(self.trigram_counts, self.bigram_counts)
        # Print sentences.
        for _ in range(n_lines):
            while next_word != "</s>":
                next_word = random.choices(
                    list(self.trigram_model[curr].keys()),
                    weights=list(self.trigram_model[curr].values()),
                    k=1,
                )[0]
                lines += [next_word]
                curr = tuple([curr[1], next_word])
            print(" ".join(lines[:-1]))
            lines = []
            next_word = ""
            curr = tuple(["<s>", "<s>"])
        # Restore previous values
        self.vocab, self.bigram_counts, self.trigram_counts, self.trigram_model = stack.pop()

    def get_simillarity(self, file):
        pp = get_perplexity(self.trigram_model, file)
        return (1 / pp)

    def fit(self, trigram_count, bigram_count):
        trigram_model = defaultdict(lambda: defaultdict(lambda: 0))
        vocab = self.vocab
        V = len(vocab)
        alpha = self.alpha

        for wi_2, wi_1 in trigram_count:
            for wi in trigram_count[tuple([wi_2, wi_1])]:
                numerator = trigram_count[(wi_2, wi_1)][wi] + alpha
                denomenator = bigram_count[wi_2][wi_1] + (alpha * V)
                trigram_model[(wi_2, wi_1)][wi] = (numerator / denomenator)

        return trigram_model


def build_lstm_model(sequence_len, n_vocab, name):
    """Build and Return an LSTM.

    :param sequence_length: Total number of sequences to train model on, size of X_train matrix.
    :param n_vocab: Total number of characters in vocabulary.
    """
    model = Sequential()
    model.add(LSTM(256, input_shape=(sequence_len, n_vocab), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(n_vocab))
    model.add(Dense(n_vocab, activation='softmax'))
    return model
