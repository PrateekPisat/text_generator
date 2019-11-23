import gensim
from gensim.models.doc2vec import LabeledSentence

import numpy as np
import os
import time
import codecs
import spacy

nlp = spacy.load('en')

#initiate sentences and labels lists
sentences = []
sentences_label = []

#create sentences function:
def create_sentences(doc):
    punctuation = [".","?","!",":","…"]
    sentences = []
    sent = []
    for word in doc:
        if word.text not in ponctuation:
            if word.text not in ("\n","\n\n",'\u2009','\xa0'):
                sent.append(word.text.lower())
        else:
            sent.append(word.text.lower())
            if len(sent) > 1:
                sentences.append(sent)
            sent=[]
    return sentences

#create sentences from files
for file_name in file_list:
    input_file = os.path.join(data_dir, file_name + ".txt")
    #read data
    with codecs.open(input_file, "r") as f:
        data = f.read()
    #create sentences
    doc = nlp(data)
    sents = create_sentences(doc)
    sentences = sentences + sents
    
#create labels
for i in range(np.array(sentences).shape[0]):
    sentences_label.append("ID" + str(i))