#%%
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
import time

def log_message(text,tab_indent=0,log_message = False):
    if log_message:
        print('\t'*tab_indent + text)


def load_data(currency,df,ticker_type ='hour', n_steps=50, scale=True, shuffle=True, window_offset=1,test_size=0.2, feature_cols=['high','low','open','close', 'volume'],log = False):
    """
    Loads binance parquet dataset
        Feature Cols: List fo features to consider for the dataset: 
            Avaliable options are:
            [   
                open_time  ,
                open  ,
                high ,
                low ,
                close ,
                volume ,
                quote_asset_volume ,
                number_of_trades ,
                taker_buy_base_asset_volume ,
                taker_buy_quote_asset_volume
            ]
            ticker_type can be of three different kinds:
            minute
            hour
            day

            since this dataframe is minute by minute trainings for the whole set would be so hard for minute to minute data

    """
    result = {}

    log_message(f'Processing {currency}',log)
    log_message(f'Loading Dataframe',1,log)

    for col in feature_cols:
        assert col in df.columns # Confirm that column is present in the dataframe
        
    if ticker_type == 'hour':
        log_message(f'Transforming dataframe to hour..',1)
        df['ticker_time'] = [ts.replace(minute=0,second=0) for ts in df.index]
        df = df.groupby('ticker_time').mean()
    elif ticker_type == 'day':
        log_message(f'Transforming dataframe to day..',1)
        df['ticker_time'] = [ts.replace(hour=0, minute=0,second=0) for ts in df.index]
        df = df.groupby('ticker_time').mean()

    result['df'] = df.copy()


    if scale:
        log_message('scaling...',1,log)
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_cols:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # Will create the labels, or target values through which to compare
    df['future'] = df['close'].shift(-window_offset)

    # Get the empty [NaN] latest values before cleaning data
    last_sequence = np.array(df[feature_cols].tail(window_offset))
    
    # drop NaNs
    log_message('cleaning...',1,log)
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)
    log_message('framing predictable futures...',log)
    for entry, target in zip(df[feature_cols].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # Here the algorithm will try to predict future prices for the dates that are not available yet in the dataframe.
    last_sequence = list(sequences) + list(last_sequence)
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    result['last_sequence'] = last_sequence
    
    log_message('Creating vector features...',1,log)
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    # Neural Network shape
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    # return the result
    log_message('Done!.',1,log)
    return result


def create_model(input_length, units=256, cell=tf.keras.layers.LSTM, n_layers=2, dropout=0.3,loss="mean_absolute_error", optimizer="rmsprop"):
    model = tf.keras.Sequential()
    for i in range(n_layers):
        if i == 0: # Create input layer
            model.add(cell(units, return_sequences=True, input_shape=(None, input_length)))
        elif i == n_layers - 1: # Create output layer
            model.add(cell(units, return_sequences=False))
        else: # Create hidden layers
            model.add(cell(units, return_sequences=True))
        model.add(tf.keras.layers.Dropout(dropout)) # add dropout after each layer
    
    model.add(tf.keras.layers.Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=[loss], optimizer=optimizer)

    return model


def save_model(model , data: pd.DataFrame, results_path: str ,currency: str):
    '''
        Function implemented to save the trained model
    '''
    file_path = os.path.join(results_path,currency) + '.h5'
    data_path = os.path.join(results_path,currency) + '.parquet'
    model.save(file_path)
    data.to_parquet(data_path)
    return file_path

def load_model(results_path: str,currency: str):
    model_path = os.path.join(results_path,currency) + '.h5'
    data_path = os.path.join(results_path,currency) + '.parquet'
    model = tf.keras.models.load_model(model_path)
    data = pd.read_parquet(data_path)
    return model,data

def plot_graphs(models,offset,window_offset):
    n_plots = len(models)
    n_rows = len(models)//2
    if n_rows==0:
        n_rows = 1
    #fig, axs = plt.subplots(2, 2)
    row = 0
    col = 0
    legend=["Actual Price", "Predicted Price"]
    x = list(range(1,offset+1))
    tit = f"Price prediction for last {offset} hours with window offset of {window_offset}"
    
    fig = plt.figure()
    ax = None
    for i in range(n_plots):
        model_set = models[i]
        currency = model_set['currency']
        data = model_set['data']
        model = model_set['model']
        
        n = len(fig.axes)
        for i in range(n):
            fig.axes[i].change_geometry(n+1, 2, i+1)

        ax = fig.add_subplot(n+1, 2, n+1)
        
        y_test, X_test, y_pred = get_plot_data(model,data)
        #ax = axs[row,col]
        ax.plot(x,y_test[-offset:],c='r',label=legend[0])
        ax.plot(x,y_pred[-offset:],c='b',label=legend[1])
        #ax.legend(loc="upper right")
        ax.set(xlabel='Days', ylabel='Price',title=currency)
        
        if i % 2 == 0:
            row +=1
            col =  0
        else:
            col +=1
            
    fig.suptitle(tit,fontsize=20)
    fig.subplots_adjust(top=0.96,hspace = 0.5 )
    
    handles, labels = ax.get_legend_handles_labels()
    
    
    fig.legend(handles,
               labels,
               loc="upper right",  # Position of legend
               borderaxespad=0.1,    # Small spacing around legend box
               title="Legend",  # Title for the legend
           )
    
           
    fig.set_size_inches(16, 10 * n_plots/2, forward=True)
    

    
    
    
    
    
def get_plot_data(model,data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["close"].inverse_transform(y_pred))
    
    
    
    return y_test, X_test, y_pred

def plot_graph(model, data,currency,offset):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["close"].inverse_transform(y_pred))
    plt.plot(y_test[-offset:], c='b')
    plt.plot(y_pred[-offset:], c='r')
    
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title(f"Price prediction for {currency} (last {offset} hours)")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def get_accuracy(model, data,window_offset):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["close"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-window_offset], y_pred[window_offset:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-window_offset], y_test[window_offset:]))
    return accuracy_score(y_test, y_pred)


def predict(model, data,n_steps, classification=False):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][:n_steps]
    # retrieve the column scalers
    column_scaler = data["column_scaler"]
    # reshape the last sequence
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
    return predicted_price


# %%
