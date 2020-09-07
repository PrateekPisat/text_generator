from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential


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
    model.add(Dense(n_vocab, activation="softmax"))
    return model
