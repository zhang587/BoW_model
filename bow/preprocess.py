'''
In this step, we preprocess the data in order to:
1. convert text to lower case
2. removed all non-word characters
3. remove all punctuations
'''

import nltk
import re
import numpy as np
import heapq

text="""
# input text here
The corrugated slopes, sheeted in mist, are clogged with jungle undergrowth and greased with mud. 
During the monsoon rains, foot trails between villages plunge again and again into gorges that hiss 
with waterfalls and fierce, impassable rivers. Navigating these natural obstacles—in a climate where 
40 feet of rainwater plummets from the sky every year—requires clever toes, iron lungs, and the power 
of prolonged observation. It demands thousands of years of attentiveness. Lifetimes of experimentation. 
Generations of problem solving.
"""
text_data = nltk.sent_tokenize(text)
for i in range(len(text_data)):
    text_data[i] = text_data[i].lower()
    text_data[i] = re.sub(r'\W', ' ', text_data[i])
    text_data[i] = re.sub(r'\s+', ' ', text_data[i])

'''
Next, we declare a dictionary to hold our bag of words
'''
word_count = {}
for data in text_data:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word_count.keys():
            word_count[word] = 1
        else:
            word_count[word] +=1

'''
We select a particular number of most frequent words
'''
freq_words = heapq.nlargest(50, word_count, key=word_count.get)


'''
The last step is to built a bag of words model
'''

X = []
for data in text_data:
    vector = []
    for word in freq_words:
        if word in nltk.word_tokenize(data):
            vector.append(1)
        else:
            vector.append(0)
    X.append(vector)
X = np.asarray(X)

print(X)