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
dropout_prob = 0.3

if tb.torch.cuda.is_available():
    device = tb.torch.device("cuda")
    print("GPU is available")
else:
    device = tb.torch.device("cpu")
    print("GPU not available, CPU used")

data = tb.SRNN_Dataset('IMDB_Dataset.csv', data_split)
#model = tb.SentimentRNN(data.vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob).to(device)
model = tb.SentimentRNN(data.vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob)
dataloader = tb.DataLoader(data, shuffle=True, batch_size=batch_size)

#Training
clips = 5
#epochs = 5
epochs = 1
lr = 0.001
loss_function = tb.nn.BCELoss()
optimizer = tb.torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    losses = []
    acc = 0.0
    model.train()
    hidden_state = model.init_hidden(batch_size, device)
    
    for tokens, labels in dataloader:
        #tokens, labels = tokens.to(device), labels.to(device)

        hidden_state = tuple([each.data for each in hidden_state])
        model.zero_grad()
        output, hidden_state = model(tokens, hidden_state, batch_size)
        breakpoint()
        break

print('Yay')
