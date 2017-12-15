from nltk import word_tokenize, pos_tag
from nltk import bigrams
from collections import defaultdict, Counter
import itertools
import pickle
import re
import random
import datetime
import sys


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

def test_sentence_combinations(sentence, model, presets):
    s = []
    for word in sentence:
        s.append(pos_tag(word_tokenize(word))[0])
    #print(s)
    
    compare = []
    for x in s:
        compare.append(x[1])
    #print(compare)
    
    list_presets = []
    for p in presets:
        if Counter(compare) == Counter(p):
            list_presets.append(p)
    #print(list_presets)
    
    best = s
    p = 0
    
    
    if len(list_presets)>0:
        temp_sent = list(s)
        for pre in list_presets:
            new_sent = []
            for pos in pre:
                for word in temp_sent:
                    if word[1] == pos:
                        new_sent.append(word)
                        break
                lis = []
                for word in temp_sent:
                    if not word[1] == pos:
                        lis.append(word)
                temp_sent = list(lis)
            new_p = probability(new_sent, model)
            if new_p > p:
                p = new_p
                best = new_sent
                #print(best)
    else:
        perms = list(itertools.permutations(s))
        for sent in perms:
            #print(sent)
            new_p = probability(sent, model)
            if new_p > p:
                p = new_p
                best = sent

    #final sentence formatting
    final_sentence = ""
    count = 0
    flag = 1
    
    if len(best)>=1:
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

def jumble_sentence(sent):
    random.shuffle(sent)
    return sent


#######################################################################

print("Start time: "+str(datetime.datetime.now()))
print()

model = []
with open("model.txt", 'rb') as f:
    model = pickle.load(f)
#print(model)
with open("test.txt", 'rb') as f:
    test = pickle.load(f)
with open("sentenceModels.txt", 'rb') as f:
    sModel = pickle.load(f)
presets = []
for k in sModel:
    temp = []
    key = re.split(" ", k)
    for w in key:
        if not w == '':
            temp.append(w)
    presets.append(temp)

total = len(test)
count = 0

i = 0
l = len(test)
byfive = []
for a in range(1,20):
    byfive.append(a*(int(l/20)))
byfive.append(l)
#print(byfive)
print("Correcting Sentences and Checking Accuracy of Corrections:")
print("--------------------100%")
for s in test:
    i += 1
    #print(i)
    if i in byfive:
        sys.stdout.write('#')
        sys.stdout.flush()
    sent = list(s)
    #print(sent)
    temp = jumble_sentence(s)
    #print(temp)
    final_sentence = test_sentence_combinations(temp, model, presets)
    new_sent = re.split(" ", final_sentence)
    n = []
    for x in new_sent:
        if not x == '':
            n.append(x)
    if n == sent:
        count += 1

accuracy = count*100/total
print()
print("Percentage correctly corrected sentences:"+str(accuracy))

#preliminary test data
'''
se = [['a', 'am', 'I', 'Dancer'], ['doctor', 'are', 'a', 'You'], ['Boy', 'plays', 'park', 'in', 'the'], ['reads', 'Mary', 'a', 'book'], ['car', 'drives', 'the', 'Sam']]
for s in se:
    print("Input sentence:")
    sent = ""
    for w in s:
        sent = sent+w+" "
    print("\t"+sent)

    print("Output sentence:")
    final_sentence = test_sentence_combinations(s, model, presets)
    print("\t"+final_sentence)
    
    print("**************************************")
'''
print()
print("End time: "+str(datetime.datetime.now()))
