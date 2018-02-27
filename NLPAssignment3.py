'''
Created on Nov 11, 2017

@author: Anand
'''

import loader
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.corpus import stopwords

path_words = "C:/Users/anand/OneDrive/University McGill/NLP/Assignment/Assignment3/Data/multilingual-all-words.en.xml"
path_key = "C:/Users/anand/OneDrive/University McGill/NLP/Assignment/Assignment3/Data/wordnet.en.key"

######################################## #########Loading Data ################################################################
if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = loader.load_instances(data_f)
    dev_key, test_key = loader.load_key(key_f)
    
    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k:v for (k,v) in dev_instances.iteritems() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.iteritems() if k in test_key}

########################################## Word Sense Disambiguation Algorithms ############################################### 

#Lists to store the calculated word senses
baseline_dev_synsets, baseline_test_synsets, lesk_dev_synsets, lesk_test_synsets = {}, {}, {}, {}


#Baseline and Lesks Algorithm on Development set
for i in dev_instances.values():
    id = i.id
    
    #Removing Stop Words
    #context = dev_instances[i].context
    #stop = set(stopwords.words('english'))
    #context = [a for a in context if a not in stop]


    base_dev_lemma = wn.lemmas(i.lemma)[0].key()
    lesk_dev_synset = lesk(i.context, i.lemma)
    
    baseline_dev_synsets[id] = base_dev_lemma
    lesk_dev_synsets[id] = lesk_dev_synset.lemmas()[0].key()
 

#Baseline and Lesks Algorithm on Test set
for i in test_instances.values():
    id = i.id
    
        
    #Removing Stop Words
    #context = dev_instances[i].context
    #stop = set(stopwords.words('english'))
    #context = [a for a in context if a not in stop]
    
    base_test_lemma = wn.lemmas(i.lemma)[0].key()
    lesk_test_synset = lesk(i.context, i.lemma)
    
    baseline_test_synsets[id] = base_test_lemma
    lesk_test_synsets[id] = lesk_test_synset.lemmas()[0].key()
    
#print(lesk_dev_synsets)
#print(baseline_dev_synsets)
########################################## Calculate Accuracy ######################################################

match_dev_base, match_test_base, match_dev_lesk, match_test_lesk = 0, 0, 0, 0

#Accuracy of Baseline Algorithm
for i in dev_key:
    if baseline_dev_synsets[i] in dev_key[i]:
        match_dev_base += 1

for i in test_key:
    if baseline_test_synsets[i] in test_key[i]:
        match_test_base += 1 

baseline_dev_accuracy = match_dev_base/(float(len(dev_key.values())))
baseline_test_accuracy = match_test_base/(float(len(test_key.values())))

print("Accuracy of Basline Algorithm on Development set: " , baseline_dev_accuracy *100)
print("Accuracy of Basline Algorithm on Test set: " , baseline_test_accuracy *100)

#Accuracy of Lesk's Algorithm    
for i in dev_key:
    if lesk_dev_synsets[i] in dev_key[i]:
        match_dev_lesk += 1

for i in test_key:
    if lesk_test_synsets[i] in test_key[i]:
        match_test_lesk += 1 

lesk_dev_accuracy = match_dev_lesk/(float(len(dev_key.values())))
lesk_test_accuracy = match_test_lesk/(float(len(test_key.values())))

print("Accuracy of Lesk's Algorithm on Development set: " , lesk_dev_accuracy *100)
print("Accuracy of Lesk's Algorithm on Test set: " , lesk_test_accuracy *100)


#################################################### Modified Algorithm #####################################################

#Modified algorithm on development set
modi_dev_synsets = {}
for i in dev_instances.values():
    id = i.id
    
    #Removing Stop Words
    #context = dev_instances[i].context
    #stop = set(stopwords.words('english'))
    #context = [a for a in context if a not in stop]
    
    base_dev_lemmas = wn.lemmas(i.lemma)
    modi_dev_synsets[id] = base_dev_lemmas[0].key()
    
    base_dev_prob = [0.7, 0.6, 0.5, 0.4, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lesk_dev_lemmas = (lesk(i.context, i.lemma)).lemmas()[0]
    lesk_dev_prob = 0.2
    max_probability = 0
    range1 = min(len(base_dev_lemmas), 5)    
    
    for x in range(0, range1):
        if(base_dev_prob[x] > max_probability):
            max_probability = base_dev_prob[x]
        if(base_dev_lemmas[x].key() == lesk_dev_lemmas.key()):
            current_prob = base_dev_prob[x] + lesk_dev_prob
            if(current_prob > max_probability):
                max_probability = current_prob
                #print(lesk_dev_lemmas.key())
                modi_dev_synsets[id] = lesk_dev_lemmas.key()
    

#Accuracy of Modified Algorithm on Development Set
modi_match_dev = 0;
for i in dev_key:
    if modi_dev_synsets[i] in dev_key[i]:
        modi_match_dev += 1
        
modi_dev_accuracy = modi_match_dev/(float(len(dev_key.values())))

print("Accuracy of Modified Algorithm on Development set: " , modi_dev_accuracy *100)


#Modified algorithm on Test set
modi_test_synsets = {}
for i in test_instances.values():
    id = i.id
    
    #Removing Stop Words
    #context = dev_instances[i].context
    #stop = set(stopwords.words('english'))
    #context = [a for a in context if a not in stop]
    
    base_test_lemmas = wn.lemmas(i.lemma)
    modi_test_synsets[id] = base_test_lemmas[0].key()
    
    base_test_prob = [0.7, 0.6, 0.5, 0.4, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lesk_test_lemmas = (lesk(i.context, i.lemma)).lemmas()[0]
    lesk_test_prob = 0.2
    max_probability = 0
    range1 = min(len(base_test_lemmas), 5)    
    
    for x in range(0, range1):
        if(base_test_prob[x] > max_probability):
            max_probability = base_test_prob[x]
        if(base_test_lemmas[x].key() == lesk_test_lemmas.key()):
            current_prob = base_test_prob[x] + lesk_test_prob
            if(current_prob > max_probability):
                max_probability = current_prob
                #print(lesk_dev_lemmas.key())
                modi_test_synsets[id] = lesk_test_lemmas.key()
    

#Accuracy of Modified Algorithm on Test Set
modi_match_test = 0;
for i in test_key:
    if modi_test_synsets[i] in test_key[i]:
        modi_match_test += 1
        
modi_test_accuracy = modi_match_test/(float(len(test_key.values())))

print("Accuracy of Modified Algorithm on Test set: " , modi_test_accuracy *100)


#################################################### Second Modified Algorithm #####################################################

#Second Modified algorithm on development set
modi_dev_synsets = {}
for i in dev_instances.values():
    id = i.id
    
    #Removing Stop Words
    #context = dev_instances[i].context
    #stop = set(stopwords.words('english'))
    #context = [a for a in context if a not in stop]
    
    base_dev_lemmas = wn.lemmas(i.lemma)
    modi_dev_synsets[id] = base_dev_lemmas[0].key()
    
    base_dev_prob = [100, 75, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0,0,0,0,0,0]
    lesk_dev_lemmas = (lesk(i.context, i.lemma)).lemmas()[0]
    lesk_dev_prob = 10
    max_probability = 0
    range1 = min(len(base_dev_lemmas), 5)    
    
    for x in range(0, range1):
        if(base_dev_prob[x] > max_probability):
            max_probability = base_dev_prob[x]
        if(base_dev_lemmas[x].key() == lesk_dev_lemmas.key()):
            current_prob = base_dev_prob[x] + lesk_dev_prob + len(i.lemma)
            if(current_prob > max_probability):
                max_probability = current_prob
                #print(lesk_dev_lemmas.key())
                modi_dev_synsets[id] = lesk_dev_lemmas.key()
    

#Accuracy of Second Modified Algorithm on Development Set
modi_match_dev = 0;
for i in dev_key:
    if modi_dev_synsets[i] in dev_key[i]:
        modi_match_dev += 1
        
modi_dev_accuracy = modi_match_dev/(float(len(dev_key.values())))

print("Accuracy of Second Modified Algorithm on Development set: " , modi_dev_accuracy *100)


#Second Modified algorithm on Test set
modi_test_synsets = {}
for i in test_instances.values():
    id = i.id
    
    #Removing Stop Words
    #context = dev_instances[i].context
    #stop = set(stopwords.words('english'))
    #context = [a for a in context if a not in stop]
    
    base_test_lemmas = wn.lemmas(i.lemma)
    modi_test_synsets[id] = base_test_lemmas[0].key()
    
    base_test_prob = [100, 75, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0,0,0,0,0,0]
    lesk_test_lemmas = (lesk(i.context, i.lemma)).lemmas()[0]
    lesk_test_prob = 0.2
    max_probability = 0
    range1 = min(len(base_test_lemmas), 5)    
    
    for x in range(0, range1):
        if(base_test_prob[x] > max_probability):
            max_probability = base_test_prob[x]
        if(base_test_lemmas[x].key() == lesk_test_lemmas.key()):
            current_prob = base_test_prob[x] + lesk_test_prob + len(i.lemma)
            if(current_prob > max_probability):
                max_probability = current_prob
                #print(lesk_dev_lemmas.key())
                modi_test_synsets[id] = lesk_test_lemmas.key()
    

#Accuracy of Second Modified Algorithm on Test Set
modi_match_test = 0;
for i in test_key:
    if modi_test_synsets[i] in test_key[i]:
        modi_match_test += 1
        
modi_test_accuracy = modi_match_test/(float(len(test_key.values())))

print("Accuracy of Second Modified Algorithm on Test set: " , modi_test_accuracy *100)
