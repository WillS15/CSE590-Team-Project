#!/usr/bin/env python3
import pandas as pd
import toolbox as tb

data_split = 0.8

embedding_dim = 64
hidden_dim = 256
num_layers = 2

#Remake init to split and shuffle
data = tb.SRNN_Dataset('IMDB_Dataset.csv', data_split)
model = tb.SentimentRNN(data.vocab_size, embedding_dim, hidden_dim, num_layers)
breakpoint()

print('Yay')
