import pandas as pd
import transformers as tran

def read_data(filename):
    '''
    Reads the data from the provided csv.
    Renames the columns: {'review': 'text', 'sentiment': 'label'}
    Renames the labels: {'positive': 1, 'negative': 0}
    '''
    
    df = pd.read_csv(filename)

    df = df.rename(columns={'review': 'text', 'sentiment': 'label'})
    df['label'] = df['label'].apply(lambda x: 1 if x=='positive' else 0)
    
    return df
    
def tokenize_data(df):
    '''
    Loads a pretrained BERT cased model
    Tokenizes the text data and replaces with PyTorch specific Tensors: ['input_ids', 'token_type_ids', 'attention_mask']
    Each Tensor can be accessed by df[<dataset_index>][<which_tensor>], ex. "df[0]['input_ids']"
    '''
    
    tokenizer = tran.AutoTokenizer.from_pretrained("bert-base-cased")

    df['text'] = df['text'].apply(str)
    df['text'] = df['text'].apply(tokenizer, padding=True, truncation=True, return_tensors="pt")

    return df
