#%%
import crypto_utils as cu
from crypto_params import *

from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import pandas as pd

def try_create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def try_create_dirs(paths):
    for path in paths:
        try_create_dir(path)

def create_tensorboard(model_name:str):
    checkpoint = ModelCheckpoint(os.path.join("results", model_name), save_weights_only=True, save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    return checkpoint, tensorboard


try_create_dirs( (RESULTS_PATH, LOG_PATH, DATA_PATH) )

def train_model(currency):
    model = cu.create_model(N_STEPS, loss=LOSS, units=N_UNITS, cell=CELL, n_layers=N_LAYERS,dropout=DROPOUT, optimizer=OPTIMIZER)

    checkpoint, tensorboard = create_tensorboard(currency)

    trained_model = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        callbacks=[checkpoint, tensorboard],
                        verbose=1)
    cu.save_model(trained_model,data, RESULTS_PATH,currency)


def test_model(currency):
    model,data = cu.load_model(RESULTS_PATH, currency )
    # model = cu.create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,dropout=DROPOUT, optimizer=OPTIMIZER)
    # model_path = os.path.join("results", model_name) + ".h5"
    # model.load_weights(model_path)

    # evaluate the model
    mse, mae = model.evaluate(data["X_test"], data["y_test"])
    # calculate the mean absolute error (inverse scaling)
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
    print("Mean Absolute Error:", mean_absolute_error)
    # predict the future price
    future_price = cu.predict(model, data)
    print(f"Future price after {WINDOW_OFFSET} days is {future_price:.2f}$")
    print("Accuracy Score:", cu.get_accuracy(model, data))
    cu.plot_graph(model, data)

def test_datamodel():

    # %%
    DATA_PATH = "D:\\Binance\\Misc\\binance-full-history"
    currencies = os.listdir(DATA_PATH)
    currencies = [x for x in currencies if '.parquet' in x]
    #%%
    currency_file = currencies[0]
    currency = os.path.splitext(currency_file)[0]
    file_path = os.path.join(DATA_PATH,currency_file)

    df = pd.read_parquet(file_path) 

    data = cu.load_data(df,'hour', N_STEPS, window_offset=WINDOW_OFFSET, test_size=TEST_SIZE, feature_cols=FEATURE_COLS)
    train_model(currencies[0])

    # %%
    import importlib
    # %%
    importlib.reload(cu)

# %%
