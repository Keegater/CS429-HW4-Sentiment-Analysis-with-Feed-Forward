import pyprind
import pandas as pd
import numpy as np
import os
import sys
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

## Load and label
# change the 'basepath' to the directory of the
# unzipped movie dataset
basepath = 'aclImdb'
labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000, stream=sys.stdout)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file),'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df._append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']

# clean
def preprocessor(text):
    text = re.sub(r'<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:>:-)?(?:\)|\(|D|P)',text)
    text = re.sub(r'\W+', ' ', text.lower())
    if emoticons:
        text = text + ' ' + ' '.join(emoticons).replace('-', '')
    return text


df['review'] = df['review'].apply(preprocessor)

# vectorize to tf-idf
count = CountVectorizer()
x_counts = count.fit_transform(df['review'])
tfidf = TfidfTransformer(use_idf=True, norm = 'l2', smooth_idf=True)
x_tfidf = tfidf.fit_transform(x_counts)


# split data
y = df['sentiment'].values
x_train, x_test, y_train, y_test = train_test_split( x_tfidf, y, test_size=0.30, random_state=42)

print("Train set:", x_train.shape, y_train.shape)
print("Test set:", x_test.shape, y_test.shape)
