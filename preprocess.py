#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# ## Parsing

# In[1]:


from pybtex.database import parse_file
from nltk.tokenize import RegexpTokenizer

import gensim

import numpy as np


# In[2]:


bib_data = parse_file('data/anthology+abstracts.bib')


# In[3]:


# check the last entry
list(bib_data.entries.keys())[-1]


# In[4]:


# number of entries in the anthology
len(list(bib_data.entries.keys()))


# In[5]:


# create raw .txt datasets for each of the past 5 years (2016-2021)
for k in bib_data.entries.keys():
    try:
        year = bib_data.entries[k].fields['year']
        abstract = bib_data.entries[k].fields['abstract']
        
        if year > '2015':
            a = open('data/datasets/abstracts.txt', 'a')
            a.write(abstract + '\n')
            a.close()
    
    # corrupted entries / entries without abstracts are skipped
    except (KeyError, UnicodeEncodeError): 
        pass


# ## Tokenization

# In[6]:


# create list of abstracts
with open('data/datasets/abstracts.txt') as f:
    text = f.read()  
    abstracts = text.split('\n')


# In[9]:


# example of abstract entry
len(abstracts)


# In[8]:


def tokenize(input_text):
    
    # makes text lowercase
    input_lower = input_text.lower()
    
    # only letters (remove numerical and special characters)
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    
    # tokenize text
    tokens = tokenizer.tokenize(input_lower)
    
    return tokens


# In[10]:


tokenized = [tokenize(a) for a in abstracts[1::20]]


# In[12]:


# example of tokenized abstract
tokenized[42]


# In[16]:


# total number of tokenized abstracts
len(tokenized)


# In[ ]:


# single token
tokenized[4][2]


# In[13]:


# create single list of tokens
tokens = [word for abstract in tokenized for word in abstract]


# In[14]:


# check single token
tokens[2021]


# In[15]:


# total number of tokens
len(tokens)

