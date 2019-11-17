from models import TrigramModel

from utils import get_test_files, get_training_files


if __name__ == "__main__":
    training_files = get_training_files()
    test_files = get_test_files()
    model = TrigramModel(training_files=training_files, alpha=1)
    # for file in test_files:
    #     print(model.get_simillarity(file))
    model.write_lines(10)
