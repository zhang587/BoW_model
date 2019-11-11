'''
In this step, we preprocess the data in order to:
1. convert text to lower case
2. removed all non-word characters
3. remove all punctuations
'''

import nltk
import re
import numpy as np

text="""
# input text here
"""
text_data = nltk.sent_tokenize(text)
for i in range(len(text_data)):
    text_data[i] = text_data[i].lower()
    text_data[i] = re.sub(r'\W', ' ', text_data[i])
    text_data[i] = re.sub(r'\s+', ' ', text_data[i])

