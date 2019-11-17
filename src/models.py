from collections import defaultdict
import random

from counts import bigram_count, trigram_count, unigram_count
from perplexity import get_perplexity


class TrigramModel:
    """Represents a trigram model."""
    def __init__(self, training_files, alpha=1):
        """ Return a trained trigram model.

        :param training_files: A list of files names which we use to train our model.
        """
        self.alpha = alpha
        self.vocab = set(unigram_count(training_files).keys())
        self.bigram_counts = bigram_count(training_files)
        self.trigram_counts = trigram_count(training_files)
        self.trigram_model = self.fit(self.trigram_counts, self.bigram_counts)

    def write_lines(self, n_lines, context=None):
        lines = []
        curr = tuple(["<s>", "<s>"])
        next_word = ""
        if not context:
            for _ in range(n_lines):
                while next_word != "</s>":
                    next_word = random.choices(
                        list(self.trigram_model[curr].keys()),
                        weights=list(self.trigram_model[curr].values()),
                        k=1,
                    )[0]
                    lines += [next_word]
                    curr = tuple([curr[1], next_word])
                next_word = ""
                curr = tuple(["<s>", "<s>"])
                print(" ".join(lines))
                lines = []
        else:
            return ""

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
                # Will handle UNK separately later.
                # if wi_1 not in vocab:
                #     wi_1 = "<UNK>"
                # if wi_2 not in vocab:
                #     wi_2 = "<UNK>"
                # if wi not in vocab:
                #     wi = "<UNK>"

                numerator = trigram_count[(wi_2, wi_1)][wi] + alpha
                denomenator = bigram_count[wi_2][wi_1] + (alpha * V)
                trigram_model[(wi_2, wi_1)][wi] = (numerator / denomenator)

        return trigram_model
