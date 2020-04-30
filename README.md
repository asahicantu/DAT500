# DAT500 Final project repository

## Authors:
* Asahi Cantu Moreno (a.cantumoreno@stud.uis.no)
* Daniel Urdal (d.urdal@stud.uis.no)

## Project title

Crypto market analysis prediction in distributed systems

## Summary

The creation of cryptocurrency and its exponential adoption over the last few years have led to the creation of many different exchanges and tools for online trading. Unlike common stock trading, cryptocurrency works nonstop all year with simple but advanced interfaces, allowing anyone with a minimum investment to enter the market and exchange assets in real time, with high liquidity and asset availability. The early state of the technology and the socioeconomic-political factors affecting the regulation of the cryptocurrency markets, as well as the speed at which transactions are made have led to a volatile high risk-high market.  This makes it very hard to analyze the market and predict prices in order to increase profits for a chosen currency. This report contains an analysis of the cryptocurrency market, and aims to generate a prediction model by analyzing historical pricing data in addition to news and events that can potentially affect the market. The models are implemented using distributed system frameworks Apache Hadoop and Apache Spark for data storage and interaction. Machine Learning models are implemented for a big data set. Further analysis, results and potential for future work are also presented in the report.

## DataSources

Due to the size of the datasets, they have not been provided in this repository, they can however be downloaded from their original sources.

* [Cryptocurency CoinTelegraph Newsfeed](https://www.kaggle.com/asahicantu/cryptocurency-cointelegraph-newsfeed)

* [Binance Full History](https://www.kaggle.com/jorijnsmit/binance-full-history)

## Repository Contents

This repository contains all the source code for the project.

- CoinTelegraph/NewsFeed/cn_content.py
    * Python script to process CoinTelegraph heading articles and save them into a single dataset.
- CoinTelegraph/NewsFeed/cn_header.py
    * Python script to process CoinTelegraph content articles and save them into a single dataset.
- CoinTelegraph/NewsFeed/KMeans_clustering.py
    * Python script to try different topic/sentiment classification mechanisms for the news dataset via clusters
- CoinTelegraph/NewsFeed/sentiment_analysis.py
    * Python script to find a general sentiment analysis for each article in the dataset
- CryptoPrediction/crypto_main.py
    * Python script to perform cyrptocurrency prediction execution and training using RNN-LSTM
- CryptoPrediction/crypto_params.py
    * Python script. Contains the default parameters created once the training model was tuned
- CryptoPrediction/crypto_utils.py
    * Python script. Contains the necessary prcedures to read cryptocurrency dataset, process it and train the ML model. Also contains plotting features
- CryptoPrediction/README.md
    * This Readme
- CryptoPrediction/requirements.txt
    * Python requirements to run the scripts
- CryptoPrediction/sprk.ipynb
    * Main file working on Hadoop distributed file system and confogured using Jupyter notebook, the connection to the cluster and settings will have to be changed to run under specific cluster conditions.
- CryptoPrediction/.ipynb_checkpoints/sprk-checkpoint.ipynb
    * Latest checkpoint of jupyter notebook file
- spark/readme.md
    * Brief description of the spark notebook.
