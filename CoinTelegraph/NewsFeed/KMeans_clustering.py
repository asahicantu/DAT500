#%%
import warnings
import tqdm
import textblob as tb
import textacy
import spacy
import re           
import re
import pandas as pd 
import os
import operator
import numpy as np  
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import math
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords   
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from contractions import contraction_mapping
from attention import AttentionLayer
from bs4 import BeautifulSoup
from spacy.lang.en.stop_words import STOP_WORDS
import string
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
extra_cols =['neg', 'neu', 'pos', 'compound', 'polarity', 'subjectivity']
for extra_col in  extra_cols:
    df[extra_col] = 0.0
#%%
items = len(df)
for i in  tqdm.tqdm(range(items)):
    data = getSentimentAnalysis(analyser,df['content'][i])
    for key in data.keys():
         df[key][i]=data[key]
#%%
punctuations = string.punctuation
stopwords = list(STOP_WORDS)
def spacy_tokenizer(sentence):
    mytokens = nlp(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

tqdm.tqdm.pandas()
df["processed_content"] = df["content"].progress_apply(spacy_tokenizer)

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
def vectorize(text, maxx_features):
    vectorizer = TfidfVectorizer(max_features=maxx_features)
    X = vectorizer.fit_transform(text)
    return X

text = df['processed_content'].values
X = vectorize(text, 2 ** 12)
X.shape
#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, random_state=42)
X_reduced= pca.fit_transform(X.toarray())
X_reduced.shape
#%%
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

# run kmeans with many different k
distortions = []
K = range(2, 50)
for k in K:
    k_means = KMeans(n_clusters=k, random_state=42).fit(X_reduced)
    k_means.fit(X_reduced)
    distortions.append(sum(np.min(cdist(X_reduced, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

X_line = [K[0], K[-1]]
Y_line = [distortions[0], distortions[-1]]

# Plot the elbow
plt.plot(K, distortions, 'b-')
plt.plot(X_line, Y_line, 'r')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
#%%
doc = nlp(df.content[1])
spacy.displacy.render(doc,style='ent')



#%%

# %%
vec = TfidfVectorizer(stop_words="english")
vec.fit(df.content.values)
features = vec.transform(df.content.values)
df['features'] = features
#%%
def plot_cluster(model,features):
    pca = PCA(n_components=2, random_state=random_state)
    reduced_features = pca.fit_transform(features.toarray())
    reduced_cluster_centers = pca.transform(model.cluster_centers_)
    plt.scatter(reduced_features[:,0], reduced_features[:,1], c=model.predict(features))
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')

random_state = 0 
clusters = range(2,10)
scores = []
for cluster in tqdm.tqdm(clusters):
    model = MiniBatchKMeans(n_clusters=5, random_state=random_state)
    model.fit(features)
    model.predict(features)
    score = silhouette_score(features, labels=model.predict(features))
    scores.append(score)
# %%
plot_cluster(model, features)

# %%
nltk.download('averaged_perceptron_tagger')
wordlemmatizer = WordNetLemmatizer()
#%%
stop_words = nlp.Defaults.stop_words
def clean_text(text):
    newString = text.lower()
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = [w for w in newString.split() if not w in stop_words]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()



#%%
clean_text(df.content[0])

# %%
from sklearn.model_selection import train_test_split
x_tr,x_val,y_tr,y_val=train_test_split(data['cleaned_text'],data['cleaned_summary'],test_size=0.1,random_state=0,shuffle=True) 

#%%
#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))

#convert text sequences into integer sequences
x_tr    =   x_tokenizer.texts_to_sequences(x_tr) 
x_val   =   x_tokenizer.texts_to_sequences(x_val)

#padding zero upto maximum length
x_tr    =   pad_sequences(x_tr,  maxlen=max_len_text, padding='post') 
x_val   =   pad_sequences(x_val, maxlen=max_len_text, padding='post')

x_voc_size   =  len(x_tokenizer.word_index) +1
#%%
from keras import backend as K 
K.clear_session() 
latent_dim = 500 

# Encoder 
encoder_inputs = Input(shape=(max_len_text,)) 
enc_emb = Embedding(x_voc_size, latent_dim,trainable=True)(encoder_inputs) 

#LSTM 1 
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True) 
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb) 

#LSTM 2 
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True) 
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1) 

#LSTM 3 
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True) 
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2) 

# Set up the decoder. 
decoder_inputs = Input(shape=(None,)) 
dec_emb_layer = Embedding(y_voc_size, latent_dim,trainable=True) 
dec_emb = dec_emb_layer(decoder_inputs) 

#LSTM using encoder_states as initial state
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) 
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c]) 

#Attention Layer
attn_layer = AttentionLayer(name='attention_layer') 
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs]) 

# Concat attention output and decoder LSTM output 
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#Dense layer
decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax')) 
decoder_outputs = decoder_dense(decoder_concat_input) 

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
model.summary()
#%%

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=50,callbacks=[es],batch_size=512, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))
#%%
from matplotlib import pyplot 
pyplot.plot(history.history['loss'], label='train') 
pyplot.plot(history.history['val_loss'], label='test') 
pyplot.legend() pyplot.show()