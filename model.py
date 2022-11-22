#!/usr/bin/env python
# coding: utf-8

import math
import random

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import gensim
from gensim.models import KeyedVectors
from gensim.models.lsimodel import LsiModel
from gensim.corpora.dictionary import Dictionary


with open('data/tokenized.txt','r') as f:
    tokenized = eval(f.read())

x_arr = np.load('data/x.npy')
y_arr = np.load('data/y.npy')

valid_split = 0.2

dataset_len = len(x_arr)
ids = list(range(dataset_len))

# length of validation set
val_len = int(np.floor(valid_split * dataset_len))
# choose random ids for validation set
val_ids = np.random.choice(ids, size=val_len, replace=False)

x_val = np.array(x_arr)[val_ids] 
y_val = np.array(y_arr)[val_ids]

x_train = np.delete(x_arr, val_ids, axis=0)
y_train = np.delete(y_arr, val_ids, axis=0)

# generator for batches
def batches(x_arr, y_arr, batch_size):
    
    pos = 0
    
    for n in range(batch_size, x_arr.shape[0], batch_size):
        x = x_arr[pos:n]
        y = y_arr[pos:n]
        pos = n
        
        yield x, y


x_data = {'train': x_train , 'val': x_val}
y_data = {'train': y_train , 'val': y_val}

# load w2v model
w2v_model = gensim.models.KeyedVectors.load('w2v.model', mmap='r')
vocab_size, emdedding_size = w2v_model.wv.vectors.shape
embedding_size = w2v_model.wv.vectors.shape[1]

# tensors to use in embedding layer
w2v_tensors = torch.FloatTensor(w2v_model.wv.vectors)

# model parameters used for LSA and PCA
n_hidden = 256
batch_size = 64
n_layers = 3

# create dictionary from tokenized abstracts
dct = Dictionary(tokenized)
# filter it for min num of words (5) and max appearance in corpus (<10%)
dct.filter_extremes(no_below=5, no_above=0.1)

# create corpus using bag of words
corpus = [dct.doc2bow(a) for a in tokenized]

# LSA model training
lsi = LsiModel(corpus, id2word=dct, num_topics=n_hidden, decay=0.2)
lsi.show_topics(5)
lsi.show_topic(3, topn=10)

trans_topics = np.transpose(lsi.projection.u)

# reduce dimensions of topic matrix
pca_topics = PCA(n_components=n_layers*batch_size, svd_solver='full').fit_transform(trans_topics)

pca_trans = np.transpose(pca_topics)

# ===== Conditioned LSTM =====

class Conditioned_LSTM(nn.Module):
    
    def __init__(self, n_hidden=256, n_layers=2, drop_prob=0.2, lr=0.001):
        super().__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.emb_layer = nn.Embedding(vocab_size, embedding_size)

        # LSTM
        self.lstm = nn.LSTM(embedding_size, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # fully-connected layer
        self.fc = nn.Linear(n_hidden, vocab_size)      
    
    def forward(self, x, hidden):
        
        x = x.long()

        # pass input through embedding layer
        embedded = self.emb_layer(x)     
        
        # get outputs and new hidden state from the lstm
        lstm_output, hidden = self.lstm(embedded, hidden)
        
        # pass through a dropout layer
        out = self.dropout(lstm_output)
        
        # flatten out
        out = out.reshape(-1, self.n_hidden) 

        # put "out" through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):

        hidden = (torch.FloatTensor(pca_trans.reshape(self.n_layers, batch_size, self.n_hidden)),
                  torch.FloatTensor(pca_trans.reshape(self.n_layers, batch_size, self.n_hidden)))

        return hidden


# instantiate the model
cond_lstm = Conditioned_LSTM(n_hidden=n_hidden, n_layers=n_layers)

print(cond_lstm)

# ===== Conditioned LSTM + Word2Vec =====

class Conditioned_LSTM_Word2Vec(nn.Module):
    
    def __init__(self, n_hidden=256, n_layers=2, drop_prob=0.2, lr=0.001):
        super().__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.emb_layer = nn.Embedding.from_pretrained(w2v_tensors)

        # LSTM
        self.lstm = nn.LSTM(embedding_size, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # fully-connected layer
        self.fc = nn.Linear(n_hidden, vocab_size)      
    
    def forward(self, x, hidden):
        
        x = x.long()

        # pass input through embedding layer
        embedded = self.emb_layer(x)     
        
        # get outputs and new hidden state from the lstm
        lstm_output, hidden = self.lstm(embedded, hidden)
        
        # pass through a dropout layer
        out = self.dropout(lstm_output)
        
        # flatten out
        out = out.reshape(-1, self.n_hidden) 

        # put "out" through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):

        hidden = (torch.FloatTensor(pca_trans.reshape(self.n_layers, batch_size, self.n_hidden)),
                  torch.FloatTensor(pca_trans.reshape(self.n_layers, batch_size, self.n_hidden)))

        return hidden


# instantiate the model
cond_lstm_w2v = Conditioned_LSTM_Word2Vec(n_hidden=n_hidden, n_layers=n_layers, drop_prob=0)

print(cond_lstm_w2v)


def train(model, epochs=10, batch_size=32, lr=0.001, clip=1, print_every=32):
    
    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # loss values
    train_loss = []
    val_loss = []
    
    # dict to assing to right loss list according to phase
    losses = {'train': train_loss , 'val': val_loss}

    for e in range(epochs):
        
        # cycle through two phases
        for phase in ['train', 'val']:
            model.train(True) if phase == 'train' else model.train(False)
            
            # batch counter
            batch = 0

            train_epoch_loss = []
            val_epoch_loss = []
            
            # dict to assing to right epoch loss list according to phase
            epoch_loss = {'train': train_epoch_loss , 'val': val_epoch_loss} 

            for x, y in batches(x_data[phase], y_data[phase], batch_size):

                batch += 1

                # initialize hidden state
                h = model.init_hidden(batch_size)

                # convert numpy arrays to PyTorch arrays
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                # detach hidden states
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                model.zero_grad()

                # get the output from the model
                output, h = model(inputs, h)

                # calculate the loss
                loss = criterion(output, targets.view(-1).long())

                if phase == 'train':
                    # back-propagate error
                    loss.backward()

                    # `clip_grad_norm` helps prevent exploding gradient
                    nn.utils.clip_grad_norm_(model.parameters(), clip)

                    # update weigths
                    opt.step()
                
                # add current batch loss to epoch loss list
                epoch_loss[phase].append(loss.item())

                # show epoch - batch - loss every n batches
                if batch % print_every == 0:

                    tot_batches = int(x_data[phase].shape[0] / batch_size)

                    print("Epoch: {}/{} -".format(e+1, epochs),
                          "Batch: {}/{} -".format(batch, tot_batches),
                          "{} loss: {:.5f}".format(phase.capitalize(), loss))
                    
            # calculate average epoch loss
            avg_epoch_loss = sum(epoch_loss[phase])/len(epoch_loss[phase])
                    
            # print average train and val loss at the end of each epoch
            print("\nEpoch: {}/{} -".format(e+1, epochs),
                  "Average {} loss: {:.5f}\n".format(phase, avg_epoch_loss))           

            # save average epoch loss for training and validation
            losses[phase].append(avg_epoch_loss)

    return train_loss, val_loss


# checkpoint save filepath
checkpoint_path = 'weights/training_checkpoints.pt'
epochs = 10

# optimizers
lstm_opt = torch.optim.Adam(cond_lstm.parameters(), lr=0.001)
lstm_w2v_opt = torch.optim.Adam(cond_lstm_w2v.parameters(), lr=0.001)

# loss criterion
criterion = nn.CrossEntropyLoss()

# save checkpoints
torch.save({
            'epoch': epochs,
            'loss': criterion,
            'lstm_state_dict': cond_lstm.state_dict(),
            'lstm_w2v_state_dict': cond_lstm_w2v.state_dict(),
            'lst_opt_state_dict': lstm_opt.state_dict(),
            'lst_opt_w2v_state_dict': lstm_w2v_opt.state_dict(),
            }, checkpoint_path)


def word_to_id(word):
    return w2v_model.wv.key_to_index[word]

def id_to_word(id):
    return w2v_model.wv.index_to_key[id]


# predict next token using top k sampling
def predict(model, top_k, t, h=None): # default value as None for first iteration
         
    # tensor inputs
    x = np.array([[word_to_id(t)]])
    inputs = torch.from_numpy(x)

    # detach hidden state from history
    h = tuple([each.data for each in h])

    # get the output of the model
    out, h = model(inputs, h)

    # get the token probabilities
    p = F.softmax(out, dim=1).data
    
    # convert to np
    p = p.numpy()
    p = p.reshape(p.shape[1],)

    # get indices of top n values
    top_ids = p.argsort()[-top_k:][::-1]

    # sample id of next word from top k values
    next_id = top_ids[random.sample(range(top_k),1)[0]]

    # return the value of the predicted word and the hidden state
    return id_to_word(next_id), h


gen_batch_size = 1

# PCA for generation
gen_pca_topics = PCA(n_components=n_layers * gen_batch_size, svd_solver='full').fit_transform(trans_topics)
gen_pca_trans = np.transpose(gen_pca_topics)

# function to generate text
def generate(model=cond_lstm, prompt='in this paper', n=10, top_k=10):
    
    model.eval()
    
    # hidden state inizialization for generation
    h = (torch.FloatTensor(gen_pca_trans.reshape(n_layers, gen_batch_size, n_hidden)),
         torch.zeros(n_layers, gen_batch_size, n_hidden))

    words = prompt.split()

    # get token and hidden state for all words in prompt
    for t in prompt.split():
        token, h = predict(model, top_k, t, h)
    
    # append words to words list
    words.append(token)

    for i in range(n-1):
        # predict subsequent token
        token, h = predict(model, top_k, words[-1], h)
        
        # append next word to word list
        words.append(token)

    return ' '.join(words)

generate()

name = {cond_lstm: 'Conditioned LSTM', cond_lstm_w2v: 'Conditioned LSTM + Word2Vec'}
loss = {cond_lstm: val_loss, cond_lstm_w2v: val_loss_w2v}


# get minimimum validation loss within a set num of epochs
def min_val_loss(model, max_epochs=100):
    return min(loss[model][:max_epochs])


# print min val loss and min perplexity for each model
for m in name.keys():
    print("Minimum validation loss for {}: {:.5f}".format(name[m], min_val_loss(m, 50)))
    print("Perplexity for model {}: {:.2f}\n".format(name[m], math.exp(min_val_loss(m, 50))))


prompts = ['in', 'in this paper', 'in this paper we present', 'in this paper we present a novel approach to']
n_list  = [5, 10, 20, 50]
k_list  = [2, 5, 10, 20]

# output counter
counter = 0

# generate output for each combination of parameters
for m in name.keys():
    for p in prompts:
        for n in n_list:
            for k in k_list:
                counter += 1
                print("=" * 180 + "\n[{}] MODEL: {}  |  PROMPT: '{}'  |  WORDS: {}  |  TOP {}".format(counter, name[m], p, n, k))
                print("=" * 180 + "\n\n {}\n" .format(generate(model=m, prompt=p, n=n, top_k=k)))