# Binance cryptocurrency market prediction using RNN-LSTM with Tensorflow

## How to run:
Code can be run and tested using two different approaches:
* Single mode:
- Requires to extract the dataset from kaggle and modifying the code to read each parquet file from the predefined path.
* Distributed mode:
- An Apache Spark-Pyspark script was configured in jupyter notebook with a haddop distributied file system- YARN. PySpark enables 
the interface for reading and saving the data model.

- Use Tensorboard tensorboard --logdir="logs" by pointing to the right logs directory
