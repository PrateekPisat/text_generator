import logging
import os

from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from utils import build_lstm_model, get_personality_files

logging.basicConfig(level=logging.INFO)


def train_lstm_model(name, train=False, path=None):
    raw_text = ""
    filepath = "char-gen-model-{epoch:02d}-{loss:.4f}.hdf5"
    training_data = get_personality_files(name)

    save_path = ".{sep}model_artifacts{sep}{name}".format(sep=os.sep, name=name)

    # read training data
    for file in training_data:
        with open(file, encoding="utf-8-sig") as f:
            text = f.read().replace(r'\ufeff', '')
            text = text.lower()
            raw_text += text

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    n_chars = len(raw_text)
    n_vocab = len(chars)
    logging.info("Total Characters: {}".format(n_chars))
    logging.info("Total Vocab: {}".format(n_vocab))
    logging.info("Chars = {}".format(chars))

    seq_length = 100
    model = build_lstm_model(seq_length, n_vocab)
    if train:
        # Prepare Training Data.
        dataX = []
        datay = []
        for i in range(0, n_chars - seq_length, 1):
            seq_in = raw_text[i: i + seq_length]
            seq_out = raw_text[i + seq_length]
            dataX.append([char_to_int[char] for char in seq_in])
            datay.append(char_to_int[seq_out])
        n_patterns = len(dataX)
        logging.info("Sequence Length: {}".format(seq_length))
        logging.info("Total Patterns: {}".format(n_patterns))

        logging.info("Vectorizing Training Data...")
        # one hot encode the output variable
        X = np_utils.to_categorical(dataX)
        y = np_utils.to_categorical(datay)
        # import pdb; pdb.set_trace()
        logging.info("X.shape = {}".format(X.shape))
        logging.info("y.shape = {}".format(y.shape))

        # define the LSTM model
        logging.info("Building Model...")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # define the checkpoint
        logging.info("Defining Checkpoint...")
        checkpoint = ModelCheckpoint(
            os.path.join(save_path, filepath),
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            period=10,
        )
        callbacks_list = [checkpoint]

        # Train Model
        logging.info("Training Model...")
        model.fit(
            X,
            y,
            epochs=300,
            batch_size=128,
            callbacks=callbacks_list,
            shuffle=True,
            validation_split=0.2,
        )
    else:
        model.load_weights(path)

    logging.info("model saved to {}".format(os.path.join(save_path, filepath)))
    return model, char_to_int, int_to_char, n_vocab, seq_length
