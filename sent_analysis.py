#!/usr/bin/env python3
import pandas as pd
import toolbox as tb

if tb.torch.cuda.is_available():
    device = tb.torch.device("cuda")
    print("GPU is available")
else:
    device = tb.torch.device("cpu")
    print("GPU not available, CPU used")

# Data
data_split = 0.8
batch_size = 50

training_data, testing_data, vocab_size = tb.read_preprocess_split_data('IMDB_Dataset.csv', data_split)
training_data, testing_data = tb.SRNN_Dataset(training_data), tb.SRNN_Dataset(testing_data)

training_dataloader = tb.DataLoader(training_data, shuffle=True, batch_size=batch_size)
testing_dataloader = tb.DataLoader(testing_data, shuffle=True, batch_size=batch_size)

# Model
embedding_dim = 512 #BERT Max Sequence Length
hidden_dim = 128
no_layers = 2
dropout_prob = 0.3 if no_layers > 1 else 0

model = tb.SentimentRNN(vocab_size, embedding_dim, hidden_dim, no_layers, dropout_prob).to(device)
print(model)

# Learning Params
clip = 5
epochs = 11
learning_rate = 0.005
loss_function = tb.nn.BCELoss()
optimizer = tb.torch.optim.Adam(model.parameters(), lr=learning_rate)

# Run History Params
training_losses = tb.np.empty(epochs)
training_accs = tb.np.empty(epochs)
testing_losses = tb.np.empty(epochs)
testing_accs = tb.np.empty(epochs)

# Running
for epoch in range(epochs):
    losses = tb.np.array([])
    accs = 0
    hidden_state = model.init_hidden(batch_size, device)

    print(f"Epoch {epoch+1}")
    model.train()    
    for tokens, labels in training_dataloader:
        tokens, labels = tokens.to(device), labels.to(device)

        hidden_state = tuple([each.data for each in hidden_state])
        model.zero_grad()
        output, hidden_state = model(tokens, hidden_state, batch_size)

        loss = loss_function(output.clamp(0, 1), labels.float())
        loss.backward()
        losses = tb.np.append(losses, [loss.item()])

        accs += tb.calc_accuracy(output, labels)

        tb.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    training_accs[epoch] = accs / len(training_dataloader.dataset)        
    training_losses[epoch] = tb.np.mean(losses)

    print(f"Training: Accuracy: {training_accs[epoch]*100}, Loss: {training_losses[epoch]}")

    losses = tb.np.array([])
    accs = 0
    hidden_state = model.init_hidden(batch_size, device)

    model.eval()
    for tokens, labels in testing_dataloader:
        tokens, labels = tokens.to(device), labels.to(device)

        hidden_state = tuple([each.data for each in hidden_state])
        output, hidden_state = model(tokens, hidden_state, batch_size)

        loss = loss_function(output.clamp(0, 1), labels.float())
        losses = tb.np.append(losses, [loss.item()])

        accs += tb.calc_accuracy(output, labels)

    testing_accs[epoch] = accs / len(testing_dataloader.dataset)        
    testing_losses[epoch] = tb.np.mean(losses)
        
    print(f"Testing: Accuracy: {testing_accs[epoch]*100}, Loss: {testing_losses[epoch]}")

breakpoint()
print('Yay')
