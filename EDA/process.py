import os
import sys

import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

import cv2

CATEGORY = 'category'
CATEGORY_ID = 'category_id'
CATEGORYTYPE = 'categorytype'
CATEGORYTYPE_ID = 'categorytype_id'
order = list(range(12))

json_file_train = os.path.join(
    '../main', 'labels',
    '{}_no_dup.json'.format('train'))


# reading json file, using dictionary oriented records, normalizing with respect to (items and set_id)
df_items = json_normalize(
    pd.read_json(json_file_train).to_dict(orient='records'),
    'items', 'set_id')


df_items = df_items\
    .rename(columns={'categoryid': 'category_id'}) # name this column as category_id


# category reading labels reading category2categorytype.tsv; what is tsv (tab separated values)
# this provides the high level category of -- outer includes -- jackets, hoodies, sweater, coats
ctg_tsv_file = os.path.join(
    '../data',
    'category2categorytype.tsv')


# drop num items from cat2cat.tsv # dictionary that tells me the category !!! 
df_ctg = pd.read_csv(ctg_tsv_file, sep='\t').drop('num items', axis=1)




# giving numbers between 0 and 11 to outers, tops, bottomss, shoes etc
df_items = df_items.merge(df_ctg, how='left', on='category_id')



df_items[df_items[CATEGORYTYPE_ID].isin(order)]


tsv_file = '../data/labels/category_id.tsv'
pd.read_csv(tsv_file, sep='\t')

category_types = [
    "outer",
    "tops",
    "full",
    "bottoms",
    "shoes",
    "legwear",
    "headwear",
    "bag",
    "accessory",
    "cosmetics",
    "_other_fashion",
    "_other",
]

df_ctgtp = pd.Series(category_types).reset_index()
df_ctgtp.columns = ['categorytype_id', 'categorytype']


arr = ['bottoms']
img_dir = '../main/images'

print("STARTED")
for item in arr:
    items = df_items[df_items['categorytype']==item]
    
    temp = []
    for i in range(items.shape[0]):

        id_num = items['set_id'].iloc[i]
        index = items['index'].iloc[i]

        image = cv2.imread(os.path.join(img_dir + '/' + str(id_num), str(index) + '.jpg'))
        temp.append(cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA))
    temp = np.array(temp)
    
    print(temp.shape)
    np.save(item+'.npy',temp)

