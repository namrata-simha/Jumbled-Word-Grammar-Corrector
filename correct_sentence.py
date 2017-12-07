from nltk import word_tokenize, pos_tag
from nltk import bigrams
from collections import defaultdict, Counter
import itertools
import pickle
import re

def probability(s, model):
    final_prob = 1
    flag = 0
    count = 0
    for w1, w2 in bigrams(s):
        count += 1
        for w in model:
            if w[0] == w1[1] and w[1] == w2[1]:
                prob = model[w]
                final_prob *= prob
                flag += 1
            else:
                prob = 1
                final_prob *= prob
        if flag < count:
            final_prob /= 1000

    if flag == 0:
        final_prob = 0
    if final_prob == 1.0:
        if flag == len(s):
            final_prob = 1.0
        else:
            final_prob = flag/(len(s)*10)

    return final_prob

def test_sentence_combinations(sentence, model):
    s = []
    for word in sentence:
        s.append(pos_tag(word_tokenize(word))[0])
    perms = list(itertools.permutations(s))
    best = s
    p = 0
    for sent in perms:
        new_p = probability(sent, model)
        if new_p > p:
            p = new_p
            best = sent

    #final sentence formatting
    final_sentence = ""
    count = 0
    flag = 1
    if best[len(best)-1][0] == '.':
        flag = 0
    for w in best:
        final_sentence += w[0]
        if flag == 1:
            final_sentence += " "
        elif not count == len(best)-2:
            final_sentence += " "
        count += 1

    return final_sentence
model = []
with open("model.txt", 'rb') as f:
    model = pickle.load(f)
#preliminary test data
print("Input sentence:")
s = ['a', 'am', 'I', 'Dancer', '.']
s = ['doctor', 'are', 'a', 'You', '.']
s = ['Boy', 'plays', 'park', 'in', 'the']
s = ['reads', 'Mary', 'a', 'book']
s = ['car', 'drives', 'the', 'Sam']
sent = ""
for w in s:
    sent = sent+w+" "
print("\t"+sent)

print("Output sentence:")
final_sentence = test_sentence_combinations(s, model)
print("\t"+final_sentence)