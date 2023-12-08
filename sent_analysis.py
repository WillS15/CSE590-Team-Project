#!/usr/bin/env python3
import pandas as pd
import toolbox as tb

#Data
data_split = 0.8
batch_size = 100

#Model
embedding_dim = 64
hidden_dim = 256
num_layers = 2

data = tb.SRNN_Dataset('IMDB_Dataset.csv', data_split)
model = tb.SentimentRNN(data.vocab_size, embedding_dim, hidden_dim, num_layers)
dataloader = tb.DataLoader(data, shuffle=True, batch_size=batch_size)
breakpoint()

print('Yay')
