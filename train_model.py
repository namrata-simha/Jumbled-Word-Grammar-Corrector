from nltk import word_tokenize, pos_tag
from nltk import bigrams
from collections import defaultdict, Counter
import itertools
import pickle
import re
import time
import sys
import datetime


def make_dataset():
    s = []
    final_s = [['My', 'name', 'is', 'Namrata'], ['I', 'am', 'an', 'Engineer'], ['I', 'am', 'a', 'Singer'], ['I', 'was', 'a', 'karate', 'player'], ['You', 'are', 'a', 'singer']]
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
        sentences = re.sub(",", "", sentences)
        sentences = re.sub("voice-over", "", sentences)
        sentences = re.sub("Voice-over", "", sentences)
        sentences = re.sub("Voice over", "", sentences)
        sentences = re.sub("Voiceover", "", sentences)
        sentences = re.sub("--", "", sentences)
        sentences = re.sub("voiceover", "", sentences)
        sentences = re.sub("voice over", "", sentences)
        sentences = re.sub("Mr[.]", "", sentences)
        sentences = re.sub("Ms[.]", "", sentences)
        sentences = re.sub("Mrs[.]", "", sentences)
        sentences = re.sub("Dr[.]", "", sentences)
        sentences = re.sub("U[.]S", "US", sentences)
        sentences = re.sub("U[.]S[.]S[.]R[.]", "USSR", sentences)
        sentences = re.split("[.]", sentences)
        temp = []
        for sen in sentences:
            temp += re.split("[?] ", sen)
        #print(temp)
        sentences = temp
        for sen in sentences:
            #print(sen)
            #new = sen+"."
            new = sen
            s.append(re.split(" ", new))

        for x in s:
            #print(x)
            if not x:
                continue
            if x == [""] or x == ["."]:
                continue
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
    i = 0
    l = len(sents)
    byfive = []
    for a in range(1,20):
        byfive.append(a*(int(l/20)))
    byfive.append(l)
    #print(byfive)
    print("--------------------100%")
    for s in sents:
        i += 1
        #print(i)
        if i in byfive:
            sys.stdout.write('#')
            sys.stdout.flush()
        new_s = []
        for word in s:
            new_s.append(pos_tag(word_tokenize(word))[0])
        new_sents.append(new_s)
        #print(new_s)
    return new_sents

def find_probs(sents):
    print()
    print()
    print("Finding Probabilities")
    print("--------------------100%")
    model = defaultdict(lambda: 0)
    for sentence in sents:
        for w1, w2 in bigrams(sentence):
            model[(w1[1], w2[1])] += 1
    
    counts = dict(Counter(model))
    probs = dict(Counter(model))
    sys.stdout.write('#')
    i = 0
    l = len(counts)
    byfive = []
    for a in range(1,19):
        byfive.append(a*(int(l/20)))
    byfive.append(l)
    #print(byfive)
    for tup in counts:
        n = 0
        i += 1
        #print(i)
        if i in byfive:
            sys.stdout.write('#')
            sys.stdout.flush()
        for w in counts:
            if w[0] == tup[0]:
                n += counts[w]
        probs[tup] = counts[tup]/n
        #print(probs[tup])

    return probs
    
def storeSentModels(sents):
    temp = []
    for s in sents:
        new_s = ""
        for w in s:
            new_s = new_s + w[1] + " "
        temp.append(new_s)
    new_sents = dict(Counter(temp))
    #print(new_sents)
    # store test data in file
    with open("sentenceModels.txt", 'wb') as f:
        pickle.dump(new_sents, f)


print("Start time: "+str(datetime.datetime.now()))
print()
#preliminary test data
#sents = [['My', 'name', 'is', 'Namrata', '.'], ['I', 'am', 'an', 'Engineer', '.'], ['I', 'am', 'a', 'Singer', '.'], ['I', 'was', 'a', 'karate', 'player', '.']]
#sents = sents+make_dataset()
sents = make_dataset()
temporary = []
for s in sents:
    if len(s) <= 7:
        temporary.append(s)
sents = temporary
l = len(sents)
train_length = int(0.6*l)
test_length = l-train_length
train = sents[:train_length]
test = sents[train_length:]
# store test data in file
with open("test.txt", 'wb') as f:
    pickle.dump(test, f)

print("Total:"+str(l))
print("Train:"+str(train_length))
print("Test:" + str(test_length))

sents = train

sents = find_pos(sents)

storeSentModels(sents)

model = find_probs(sents)

#store model in file
with open("model.txt", 'wb') as f:
    pickle.dump(model, f)

print()
print()
print("End time: "+str(datetime.datetime.now()))
