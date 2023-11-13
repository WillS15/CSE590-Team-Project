import pandas as pd
import torch
import torch.nn as nn
#from nltk.tokenize import word_tokenize
#import nltk
from transformers import BertTokenizerFast

class SRNN_Dataset():
    def __init__(self, filename, split_val):

        print('Reading and Preprocessing ... ', end='', flush=True)
        
        df = pd.read_csv(filename)
        df = df.rename(columns={'review': 'text', 'sentiment': 'label'})

        #Replace 'positive' and 'negative' labels with 1 or 0 respectively
        df['label'] = df['label'].map(lambda x: 1 if x=='positive' else 0)

        print('Done')
        print('Cleaning ... ', end='', flush=True)

        def clean_sent(sent):
            import re
            #HTML Tags
            sent = re.sub(r"(<.*?>){1,2}", " ", sent)
            
            #Nonword characters
            #sent = re.sub(r"[^\w\s]", "", sent)

            #Removes repeating whitespace
            sent = re.sub(r"\s+", " ", sent)

            return sent
            
        df['text'] = df['text'].apply(clean_sent)
        #breakpoint()
        
        print('Done')
        print('Tokenizing ... ', end='', flush=True)

        #nltk.download('punkt')
        #df['text'] = df['text'].apply(word_tokenize)
        
        #BERT Tokenizer & Encoding
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        df = pd.concat([df['text'], df['text'].apply(tokenizer.encode, padding=True, truncation=True, return_tensors="pt").rename('tokens'), df['label']], axis=1)

        print('Done')
        print('Shuffle and Split ... ', end='', flush=True)

        df = df.sample(frac=1, ignore_index=True, random_state=1)
        split_index = round(len(df)*split_val)
        train, test = df[:split_index].reset_index(drop=True), df[split_index:].reset_index(drop=True)

        print('Done')

        self.train = train
        self.test = test
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return len(self.train.index) + len(self.test.index)

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(SentimentRNN, self).__init__()

        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.out = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, sent):
        pass
