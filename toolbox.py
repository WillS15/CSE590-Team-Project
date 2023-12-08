import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#from nltk.tokenize import word_tokenize
#import nltk
from transformers import BertTokenizerFast

class SRNN_Dataset(Dataset):
    """
    Takes the entire dataset from a csv and performs preprocessing and splitting.
    After splitting, this class works as a PyTorch dataset, intended to be used with a DataLoader.
    The required methods for a DataLoader will work with the training data only.
    Test data will be stored in this class but will need to be accessed directly.
    """
    
    def __init__(self, filename, split_val=0.8):
        """
        Arguments:
        
        filename (string)
        split_val (float): Percentage of train test split
        """

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
        df = pd.concat([df['text'], df['text'].apply(tokenizer.encode, padding="max_length", truncation=True, return_tensors="pt").rename('tokens'), df['label']], axis=1)
        df.tokens = df.tokens.map(torch.flatten)

        print('Done')
        print('Shuffle and Split ... ', end='', flush=True)

        df = df.sample(frac=1, ignore_index=True, random_state=1)
        split_index = round(len(df)*split_val)
        train, test = df[:split_index].reset_index(drop=True), df[split_index:].reset_index(drop=True)

        print('Done')

        self.split_val = split_val
        self.train = train
        self.test = test
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return len(self.train.index)

    def __getitem__(self, ind):
        dic = {"text": self.train.iloc[ind].text, "tokens": self.train.iloc[ind].tokens, "label": self.train.iloc[ind].label}
        return self.train.iloc[ind].to_dict()

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(SentimentRNN, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("GPU is available")
        else:
            self.device = torch.device("cpu")
            print("GPU not available, CPU used")

        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.out = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, sent):
        pass
