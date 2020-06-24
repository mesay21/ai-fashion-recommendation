from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


import sys

import pandas as pd
from pandas.io.json import json_normalize


json_file = '../data/labels/train_no_dup.json'
df = pd.read_json(json_file)
print(df)


print(df['items'].iloc[0])

print(len(set(df['name'])))
print(set(df['name']))


print(df['items'].iloc[0])

json_file = '../main/labels/test_no_dup.json'
test_df = pd.read_json(json_file)
test_df


print(test_df.name.value_counts()[:40])


test_df.name.value_counts()[:40]


dictionary = {}
for i in range(df.shape[0]):
    length = len(df['items'].iloc[i])
    if length not in dictionary:
        dictionary[length] = 1
    else:
        dictionary[length] += 1
        
print(dictionary)


json_file = '../main/labels/test_no_dup.json'
df = pd.read_json(json_file)
df
dictionary = {}
for i in range(df.shape[0]):
    length = len(df['items'].iloc[i])
    if length not in dictionary:
        dictionary[length] = 1
    else:
        dictionary[length] += 1
        
print(dictionary)