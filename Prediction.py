
#%%
import numpy as np # working with data
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime
import os

#dir_path = os.getcwd()
#%%
def get_date_from_current(offset):
    return datetime.datetime.now().date() - datetime.timedelta(days=offset)


def create_data_folder():
    if not os.path.exists("data"):
        os.makedirs("data")


#%%
binanceDF = pd.read_csv('..\\DataSets\\Binance.csv')
binanceDF.drop(binanceDF.columns[0],axis=1,inplace=True)
#%%
bnb = binanceDF[binanceDF.Currency == 'BNB-BTC']
#%%
currency_close_price = currency_data.close.values.astype('float32')
currency_close_price = currency_close_price.reshape(len(currency_close_price), 1)

#%%
def create_datasets(dataset, sequence_length):
    sequence_length += 1
    seq_dataset = []
    for i in range(len(dataset) - sequence_length):
        seq_dataset.append(dataset[i: i + sequence_length])

    seq_dataset = np.array(seq_dataset)
    
    data_x = seq_dataset[:, :-1]
    data_y = seq_dataset[:, -1]
        
    return data_x, data_y

scaler = MinMaxScaler(feature_range=(0, 1))
currency_close_price_scaled = scaler.fit_transform(currency_close_price)

train_size = int(len(currency_close_price_scaled) * 0.85)
test_size = len(currency_close_price_scaled) - train_size
train, test = currency_close_price_scaled[0:train_size,:], currency_close_price_scaled[train_size:len(currency_close_price_scaled),:]

look_back = 10

x_train, y_train = create_datasets(train, look_back)
x_test, y_test = create_datasets(test, look_back)


#%%
model = Sequential()

model.add(LSTM(
    input_shape=(None,1),
    units=50,
    return_sequences=True))
model.add(Dropout(0.35))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.35))

model.add(Dense(
    units=1))
model.add(Activation('linear'))

model.compile(loss='mse', optimizer='rmsprop')

history = model.fit(x_train, y_train, batch_size=64, epochs=30, verbose=2, validation_split=0.2)

#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#%%
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

train_predict_unnorm = scaler.inverse_transform(train_predict)
test_predict_unnorm = scaler.inverse_transform(test_predict)

# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
trainPredictPlot = np.empty_like(currency_close_price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict_unnorm)+look_back, :] = train_predict_unnorm

# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
testPredictPlot = np.empty_like(currency_close_price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict_unnorm)+(look_back*2)+1:len(currency_close_price)-1, :] = test_predict_unnorm

plt.figure(figsize=(19, 10))
plt.plot(currency_close_price, 'g', label = 'original dataset')
plt.plot(trainPredictPlot, 'r', label = 'training set')
plt.plot(testPredictPlot, 'b', label = 'predicted price/test set')
plt.legend(loc = 'upper left')
plt.xlabel('Time in Days')
plt.ylabel('Price')

plt.title("%s price %s - % s" % (currency, 
                                 utilities.get_date_from_current(offset=len(currency_close_price)), 
                                 utilities.get_date_from_current(0)))

plt.show()