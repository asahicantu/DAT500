#%%
import tqdm
import textblob as tb
import textacy
import spacy
import re
import pandas as pd 
import os
import operator
import numpy as np  
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import math
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#%%
#pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
#python -m spacy download en_trf_xlnetbasecased_lg
# pip install vaderSentiment
# pip install textacy
#nlp = spacy.load('en_trf_xlnetbasecased_lg')
#https://github.com/thushv89/attention_keras/blob/master/layers/attention.py
#%%
def getSentimentAnalysis(analyzer,text):
    tb_text_sentiment = tb.TextBlob(text).sentiment
    vd_text_sentiment = analyser.polarity_scores(text)
    vd_text_sentiment['polarity'] = tb_text_sentiment.polarity
    vd_text_sentiment['subjectivity'] = tb_text_sentiment.subjectivity
    return vd_text_sentiment
#%%
analyser = SentimentIntensityAnalyzer()
df = pd.read_csv("D:\Binance\DataSets\cointelegraph_news_content.csv")
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
nlp = spacy.load('en_core_web_lg')
#%%
# Initialize sentiment analysis columns
sent_cols =['neg', 'neu', 'pos', 'compound', 'polarity', 'subjectivity']
for extra_col in  sent_cols:
    df[extra_col] = 0.0
#%%
items = len(df)
for i in  tqdm.tqdm(range(items)):
    data = getSentimentAnalysis(analyser,df['content'][i])
    for key in data.keys():
         df[key][i]=data[key]
#%%
punctuations = string.punctuation
stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)
def spacy_tokenizer(sentence):
    mytokens = nlp(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

tqdm.tqdm.pandas()
df["processed_content"] = df["content"].progress_apply(spacy_tokenizer)


# %%
