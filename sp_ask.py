# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 08:17:54 2018

@author: HP
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('C://Users/HP/Desktop/software_project/intents1.json') as json_data:
    intents = json.load(json_data)
    
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load('C://Users/HP/Desktop/software_project/f_my_model')

#input sentence clean up

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

#input sentence converted to tensors, to be fed into model
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):  #if w=['a','b','c'];enumerate(w)=[(0,'a'),(1,'b'),(2,'c')]
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))
    
    
p = bow("is your shop open today?", words)
print (p)
print (classes)

#tf.reset_default_graph()  


 
#we need to save training_data file because model will only give us the response corresponding to query ,
#but if we wnt to know about the classes involved we need the training_data file




# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    #since we are doing regression and not classification here, we are not getting on eclasss, we are getting a group 
    #of classes,of which one could be the required class
    
    #so, we need to filter out the most relevant classes, or classes whose probability is greater than 25%

    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    #return results
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    print(results)
    res=[]
    #print(results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        for g in results:
            for i in intents['intents1']:
                # find a tag matching the first result
                if i['tag'] == g:
                    res.append(i['responses'])
                    # set context for this intent if necessary
                    #return(i['responses'])
                    '''
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return print(random.choice(i['responses']))

            results.pop(0)
            '''
    return res
         

#classify('is your shop open today?')
response('hi')
#response('do you take cash?')
