from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def get_summary(text):

    freq_table = create_frequency_table(text)
    sents = sent_tokenize(text)
    sentence_scores = score_sentences(sents, freq_table)
    threshold = find_average_score(sentence_scores)
    summary = generate_summary(sents, sentence_scores, 1.5 * threshold)
    return summary


def create_frequency_table(text_string):
    words = word_tokenize(text_string)
    freq_table = dict()
    for word in words:
        if word not in stop_words:
            freq_table[word] = freq_table.get(word, 0) + 1
    return freq_table


def score_sentences(sentences, freq_table):
    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        for wordValue in freq_table:
            if wordValue in sentence.lower():
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freq_table[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freq_table[wordValue]

        sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence

    return sentenceValue


def find_average_score(sentenceValue):
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))
    return average


def generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary
