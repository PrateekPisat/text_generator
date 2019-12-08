# Text Generator

## Setup

To install all dependancies:

```bash
    pip install -r ./deps/requirements.in
```

## LSTM Models

### Build LSTM Model

There are two approachs to build a model:

```bash
    cd src
    python text_generator.py --build_model <dir name> <num epochs>
    cd ..
```

Here `dir_name` is the name where the text corpus is saved, and num_epochs is then number of epochs to which the model will be trained.<br>
**NOTE**: this directory should be located as the following path:
`./src/train` <br>
For Example: If the text corpus is stored in directory name: `lincoln/`, then copy the contenets of the directory to the path `./src/train/lincoln/` and then call

```bash
    python text_generator.py --build_model lincoln 100
```

The other apporach is to use the provided Makefile. However, by deafult the makefile is designed to only work with Abraham Lincoln's speehes. Edit the Makefile to update this to your liking.

```bash
    make build-LSTM
```

After the model has been built, it will be saved to `./src/model_artifacts/<dir_name>`

### Generate Sentences

***NOTE***: Needs a saved model.

To generate sentences using a saved model you can either:

```bash
    cd src
    python text_generator.py --generate_sent <name> <n_chars>  <"Seed Text"> <path/to/saved/model>
    cd ..
```

Here, `name` should be the same directory name as provided in the model building step. This is important for generating sentences. `n_chars` is the number of characters you want to generate. `Seed Text` is the seed sentence/s that will be used as context for generating sentences. **Note** The seed text should at least be 100 characters long. `<path/to/saved/model>` is where you saved you build model.

The other approach is to use the makefile. However, by deafult the makefile is designed to only work with Abraham Lincoln's model. Edit the Makefile to update this to your liking.

```bash
    make run-LSTM
```

We have provided a few pre-built models to test. They are stored in `./src/model_artifacts`, where each folder contains a model representing that author.

### Test Similarity

***NOTE***: Needs a saved model.

One method to test the similarity of some text with the writing style of the model(Author) is :

```bash
    cd src &&
    python text_generator.py --similarity <name> "<Test Text>" <path/to/saved/model>
    && cd .
```

Here, `name` should be the same directory name as provided in the model building step. `Test Text` is the text that you want to tes the similarity of. **Note** The test text should at least be 100 characters long. `<path/to/saved/model>` is where you saved you build model.

## Trigram Model

### Generate Sentences

To generate sentences using the trigram model:

```bash
    cd src
    python text_generator.py --trigram-generate <name> <n_words>
    cd ..
```

Here, `name` should be directory name of the directory that consists of the text corpus. `n_chars` is the number of characters you want to generate.

### Test Similarity

To test the similarity of a given file using the tirgram model use:

```bash
    cd src
    python text_generator.py --trigram-similarity <name> <path/to/file_to_text>
    cd ..
```

Here, `name` should be directory name of the directory that consists of the text corpus. `<path/to/file_to_text>` is the path to the text file that you want to test for similarity.