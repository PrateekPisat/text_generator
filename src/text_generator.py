import os

from models import TrigramModel
from utils import get_personality_files, get_test_files, get_training_data


if __name__ == "__main__":
    filename = ".{sep}models{sep}trigram_model".format(sep=os.sep)
    charles = get_personality_files()
    training_data = get_training_data()
    test_files = get_test_files()
    model = TrigramModel(training_data=training_data, personality=charles)
    for file in test_files:
        print("SimScore = {}".format(model.get_simillarity(file)))
    model.write_lines(10)
