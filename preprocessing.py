#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# ### Parsing

# In[1]:


from pybtex.database import parse_file

from nltk.tokenize import RegexpTokenizer


# In[2]:


bib_data = parse_file('data/anthology+abstracts.bib')


# In[3]:


# check the last entry
list(bib_data.entries.keys())[-1]


# In[4]:


# number of entries in the anthology
len(list(bib_data.entries.keys()))


# In[24]:


# create raw .txt datasets for each of the past 5 years (2016-2021)
for k in bib_data.entries.keys():
    try:
        year = bib_data.entries[k].fields['year']
        abstract = bib_data.entries[k].fields['abstract']
        
        if year > '2015':
            y = open('data/datasets/abstracts_%s.txt' %year, 'a')
            y.write(abstract + '\n')
            
            a = open('data/datasets/all.txt', 'a')
            a.write(abstract + '\n')
            
            y.close()
            a.close()
    
    # corrupted entries / entries without abstracts are skipped
    except (KeyError, UnicodeEncodeError): 
        pass


# ### Tokenization

# In[25]:


# create list of abstracts
with open('data/datasets/all.txt') as f:
    text = f.read()  
    abstracts = text.split('\n')


# In[30]:


# example of abstract entry
abstracts[42]


# In[70]:


def tokenize(input_text):
    
    # makes text lowercase
    input_lower = input_text.lower()
    
    # only letters (remove numerical and special characters)
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    
    # tokenize text
    tokens = tokenizer.tokenize(input_lower)
    
    return tokens


# In[71]:


tokenized = [tokenize(a) for a in abstracts]


# In[72]:


# example of tokenized abstract
tokenized[42]


# In[73]:


# total number of tokenized abstracts
len(tokenized)


# In[74]:


# single token
tokenized[4][2]


# In[78]:


# create single list of tokens
tokens = [token for abstract in tokenized for token in abstract]


# In[80]:


# check single token
tokens[2021]


# In[79]:


# total number of tokens
len(tokens)

