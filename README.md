# DAT500 Final proyect repository

## Authors:
* Asahi Cantu Moreno (a.cantumoreno@stud.uis.no)
* Daniel Urdal (d.urdal@stud.uis.no)

## Project title:

Crypto market analysis prediction in distributed systems

## Summary

The creation of cryptocurrency and its massive adoption throughout the last three years has led to the emission of many different exchange and tools to trade it via Internet. Unlike common stock trading, cryptocurrency works nonstop all the year with simple and advanced interfaces, allowing any user with a minimum amount of money to enter the market and change the currency on real time due to its high liquidity and asset availability. Because of the early state of the technology and the socioeconomical-political factors affecting the regulation of the cryptocurrency markets as well as the speed at which transactions are made has led to a high risk-high volatility market, making very hard to analyze or predict prices that can lead to profits for a chosen currency. This academic report contains a deep analysis on crypto market and aims to generate a prediction model by analyzing historical data, the news and events that can potentially affect the market and its impact on the prices by using distributed systems Hadoop® and Apache Spark® for data storage and interaction. Machine Learning models are implemented as well for a big dataset. Further analysis, results and potential for future work is presented in this report
Final project repository for the project Data I

## DataSources

Due to the size of the datasets, they have not been provided in this repository, they can be downloaded however from their original sources.

* [Cryptocurency CoinTelegraph Newsfeed](https://www.kaggle.com/asahicantu/cryptocurency-cointelegraph-newsfeed)

* [Binance Full History](https://www.kaggle.com/jorijnsmit/binance-full-history)

## Repository Contents

This repository contains all the researcha and algorithms to properly classify candlestick minute data.



- D:\Binance\DAT500\CoinTelegraph\NewsFeed\cn_content.py
    * Python script to process CoinTelegraph heading articles and save them into a single dataset.
- D:\Binance\DAT500\CoinTelegraph\NewsFeed\cn_header.py
    * Python script to process CoinTelegraph content articles and save them into a single dataset.
- D:\Binance\DAT500\CoinTelegraph\NewsFeed\KMeans_clustering.py
    * Python script to try different topic/sentiment classification mechanisms for the news dataset via clusters
- D:\Binance\DAT500\CoinTelegraph\NewsFeed\sentiment_analysis.py
    * Python script to find a general sentiment analysis for each article in the dataset
- D:\Binance\DAT500\CryptoPrediction\crypto_main.py
    * Python script to perform cyrptocurrency prediction execution and training using RNN-LSTM
- D:\Binance\DAT500\CryptoPrediction\crypto_params.py
    * Python script. Contains the default parameters created once the training model was tuned
- D:\Binance\DAT500\CryptoPrediction\crypto_utils.py
    * Python script. Contains the necessary prcedures to read cryptocurrency dataset, process it and train the ML model. Also contains plotting features
- D:\Binance\DAT500\CryptoPrediction\README.md
    * This Readme
- D:\Binance\DAT500\CryptoPrediction\requirements.txt
    * Python requirements to run the scripts
- D:\Binance\DAT500\CryptoPrediction\sprk.ipynb
    * Main file working on Hadoop distributed file system and confogured using Jupyter notebook, the connection to the cluster and settings will have to be changed to run under spceific cluster conditions.
- D:\Binance\DAT500\CryptoPrediction\.ipynb_checkpoints\sprk-checkpoint.ipynb
    * Latest checkpoint of jupyter notebook file
- D:\Binance\DAT500\FinalReport\DAT500_Project_Report.pdf
    * Final project report with all the research and conclusions from the development of this project
- D:\Binance\DAT500\spark\readme.md
    * Brief description of the spark notebook.
