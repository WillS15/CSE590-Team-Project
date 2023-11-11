#!/usr/bin/env python3

import pandas as pd
import torch
import toolbox as tb

SEED = 1024

df = tb.read_data('IMDB_Dataset.csv')
df = tb.tokenize_data(df)

breakpoint()

print('Yay')
