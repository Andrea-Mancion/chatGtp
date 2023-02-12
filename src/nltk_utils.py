import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence): #tokenize a string "hello wolrd!" -> ["hello" "world" "!"]
    return nltk.word_tokenize(sentence)

def stem(word): # group linked word in one, "organized", "organization", "organize" -> organ
    return stemmer.stem(word.lower()) # this function allows to avoid duplicate words and to do something more concrete with (sorted(set)...)

def bag_of_words(tok_sentence, words_in_a_pattern): # look for all words if one of them are in all possibilities of a pattern and then if one of them are in we got 1 else 0

    tokenize_stem_sentence = [stem(word) for word in tok_sentence] # we group all the element
    prob = np.zeros(len(words_in_a_pattern), dtype=np.float32)

    for id, word in enumerate(words_in_a_pattern):
        if word in tokenize_stem_sentence: # if we find a existing word in our sentence
            prob[id] = 1.0
    return prob