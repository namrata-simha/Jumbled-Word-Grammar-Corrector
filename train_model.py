from nltk import word_tokenize, pos_tag
from nltk import bigrams
from collections import defaultdict, Counter
import itertools
import pickle
import re

def make_dataset():
    s = []
    final_s = []
    for i in range(1990, 2013):
        filename = "spoken_eng/w_spok_" + str(i) + ".txt"
        #filename = "spoken_eng/w_spok_1990.txt"
        file = open(filename, "r")
        sentences = file.read()
        # print(sentences)
        sentences = re.sub(" 's", "\'s", sentences)
        sentences = re.sub(" 'd", "\'d", sentences)
        sentences = re.sub(" 'm", "\'m", sentences)
        sentences = re.sub(" n't", "n\'t", sentences)
        sentences = re.sub("##[0-9]{6}", "", sentences)
        sentences = sentences.replace("@ @ @ @ @ @ @ @ @ @ ", "")
        sentences = sentences.replace(" @ @ @ @ @ @ @ @ @ @", "")
        sentences = re.sub("@[a-zA-Z!-]+ ", "", sentences)
        sentences = re.sub("voice over [a-zA-Z]+ :", "", sentences)
        sentences = re.sub("voice over [a-zA-Z]+ ,", "", sentences)
        sentences = re.sub("voice-over [a-zA-Z]+ :", "", sentences)
        sentences = re.sub("voice-over [a-zA-Z]+ ,", "", sentences)
        sentences = re.sub(" : ", " ", sentences)
        sentences = re.sub("@", "", sentences)
        sentences = re.sub("voice-over", "", sentences)
        sentences = re.sub("voice over", "", sentences)
        sentences = re.sub("Mr[.]", "", sentences)
        sentences = re.sub("Ms[.]", "", sentences)
        sentences = re.sub("Mrs[.]", "", sentences)
        sentences = re.sub("Dr[.]", "", sentences)
        sentences = re.sub("U[.]S", "US", sentences)
        sentences = re.sub("U[.]S[.]S[.]R[.]", "USSR", sentences)
        sentences = re.split("[.]", sentences)
        for sen in sentences:
            #print(sen)
            new = sen+"."
            s.append(re.split(" ", new))
        for x in s:
            words = []
            for word in x:
                if word == "\n":
                    continue
                elif word == "":
                    continue
                else:
                    words.append(word)
            final_s.append(words)

    #print(final_s)
    return final_s

def find_pos(sents):
    print("Finding POS Tags")
    new_sents = []
    #print(sents)
    for s in sents:
        new_s = []
        for word in s:
            new_s.append(pos_tag(word_tokenize(word))[0])
        new_sents.append(new_s)
        #print(new_s)
    return new_sents

def find_probs(sents):
    print("Finding Probabilities")
    model = defaultdict(lambda: 0)
    for sentence in sents:
        for w1, w2 in bigrams(sentence):
            model[(w1[1], w2[1])] += 1

    counts = dict(Counter(model))
    probs = dict(Counter(model))

    for tup in counts:
        n = 0
        for w in counts:
            if w[0] == tup[0]:
                n += counts[w]
        probs[tup] = counts[tup]/n
        #print(probs[tup])

    return probs

#preliminary test data
sents = [['My', 'name', 'is', 'Namrata', '.'], ['I', 'am', 'an', 'Engineer', '.'], ['I', 'am', 'a', 'Singer', '.'], ['I', 'was', 'a', 'karate', 'player', '.']]
sents = sents+make_dataset()
sents = find_pos(sents)
model = find_probs(sents)

#store model in file
with open("model.txt", 'wb') as f:
    pickle.dump(model, f)
