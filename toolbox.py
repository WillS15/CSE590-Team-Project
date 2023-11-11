import pandas as pd

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
    
