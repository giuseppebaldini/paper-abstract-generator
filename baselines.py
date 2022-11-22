#!/usr/bin/env python
# coding: utf-8

import math
import random

import numpy as np

import gensim

from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GRU, LSTM, Activation, Dense, Dropout, Embedding
from tensorflow.keras.callbacks import History

import matplotlib.pyplot as plt


# load word2vec model
w2v_model = gensim.models.KeyedVectors.load("w2v.model", mmap='r')

vocab_size, emdedding_size = w2v_model.wv.vectors.shape

x = np.load('data/x.npy')
y = np.load('data/y.npy')[:,0]

# instantiate history to save losses
history = History()


# ===== GRU =====

gru = Sequential()

gru.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size))
gru.add(GRU(128, input_shape=(vocab_size, emdedding_size), return_sequences=True))
gru.add(Dropout(0.3))
gru.add(GRU(128))
gru.add(Dense(vocab_size, activation='softmax'))

gru.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

filepath = "weights/gru.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]

gru_loss = gru.fit(x, y, validation_split=0.2, batch_size=64, epochs=20, callbacks=callbacks)

# ===== GRU + Word2Vec =====

gru_w2v = Sequential()

gru_w2v.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[w2v_model.wv.vectors]))
gru_w2v.add(GRU(128, input_shape=(vocab_size, emdedding_size), return_sequences=True))
gru_w2v.add(Dropout(0.3))
gru_w2v.add(GRU(128))
gru_w2v.add(Dense(vocab_size, activation='softmax'))

gru_w2v.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

filepath = "weights/gru_w2v.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]

gru_w2v_loss = gru_w2v.fit(x, y, validation_split=0.2, batch_size=64, epochs=20, callbacks=callbacks)

# ===== LSTM =====

lstm = Sequential()

lstm.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size))
lstm.add(LSTM(256, input_shape=(vocab_size, emdedding_size), return_sequences=True))
lstm.add(Dropout(0.3))
lstm.add(LSTM(256, return_sequences=True))
lstm.add(Dropout(0.3))
lstm.add(LSTM(128))
lstm.add(Dropout(0.3))
lstm.add(Dense(vocab_size, activation='softmax'))

lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

filepath = "weights/lstm.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]

lstm_loss = lstm.fit(x, y, validation_split=0.2, batch_size=64, epochs=20, callbacks=callbacks)

# ===== LSTM + Word2Vec =====

lstm_w2v = Sequential()

lstm_w2v.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[w2v_model.wv.vectors]))
lstm_w2v.add(LSTM(256, input_shape=(vocab_size, emdedding_size), return_sequences=True))
lstm_w2v.add(Dropout(0.3))
lstm_w2v.add(LSTM(256, return_sequences=True))
lstm_w2v.add(Dropout(0.3))
lstm_w2v.add(LSTM(128))
lstm_w2v.add(Dense(vocab_size, activation='softmax'))

lstm_w2v.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

filepath = "weights/lstm_w2v.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]

lstm_w2v_loss = lstm_w2v.fit(x, y, validation_split=0.2, batch_size=64, epochs=20, callbacks=callbacks)


def word_to_id(word):
    return w2v_model.wv.key_to_index[word]

def id_to_word(id):
    return w2v_model.wv.index_to_key[id]


# using top k sampling
def sample(preds, top_k):
    
    top_ids = preds.argsort()[-top_k:][::-1]
    next_id = top_ids[random.sample(range(top_k),1)[0]]
    
    return next_id


def generate(model=gru, prompt='In this paper', n=20, top_k=10):
    
    # split and make lowercase words from input prompts
    word_ids = [word_to_id(word) for word in prompt.lower().split()]
    
    for i in range(n):
        # get prediction using chosen model
        prediction = model.predict(x=np.array(word_ids))
        # find id of next word
        id = sample(prediction[-1], top_k)
        # append to list of output ids
        word_ids.append(id)
        
    # convert ids to words     
    words = [id_to_word(w) for w in word_ids]
    
    return ' '.join(words)


generate()


model_loss = {gru_loss: 'GRU', gru_w2v_loss: 'GRU + Word2Vec', lstm_loss: 'LSTM', lstm_w2v_loss: 'LSTM + Word2Vec'}

# get minimimum validation loss within a set num of epochs
def min_val_loss(model, max_epochs=20):
    return min(model.history['val_loss'][:max_epochs])

# print min val loss and min perplexity for each model
for m in model_loss.keys():
    print("Minimum validation loss for {}: {:.5f}".format(model_loss[m], min_val_loss(m)))
    print("Perplexity for model {}: {:.2f}\n".format(model_loss[m], math.exp(min_val_loss(m))))

models  = {gru: 'GRU', gru_w2v: 'GRU + Word2Vec', lstm: 'LSTM', lstm_w2v: 'LSTM + Word2Vec'}
prompts = ['in', 'in this paper', 'in this paper we present', 'in this paper we present a novel approach to']
n_list  = [5, 10, 20, 50]
k_list  = [2, 5, 10, 20]

# output counter
counter = 0

# generate output for each combination of parameters
for m in models.keys():
    for p in prompts:
        for n in n_list:
            for k in k_list:
                counter += 1
                print("=" * 180 + "\n[{}] MODEL: {}  |  PROMPT: '{}'  |  WORDS: {}  |  TOP {}".format(counter, models[m], p, n, k))
                print("=" * 180 + "\n\n {}\n" .format(generate(model=m, prompt=p, n=n, top_k=k)))