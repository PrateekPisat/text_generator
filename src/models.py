from collections import defaultdict

from counts import unigram_count, bigram_count, trigram_count
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
