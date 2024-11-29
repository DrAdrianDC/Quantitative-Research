# Stock Price Prediction Using Markov Chains and Machine Learning


## Overview

This project demonstrates the use of Markov Chains and Machine Learning for predicting stock price movements. By analyzing historical stock data, we classify daily price movements into discrete states (Up, Down, Stable) and model the transitions between these states using a Markov Chain. Additionally, we train a machine learning model to compare its predictive performance against the Markov model.



## Motivation

Predicting stock prices is a challenging task due to the inherent stochastic nature of financial markets. This project explores a hybrid approach, combining probabilistic Markov models with machine learning, to gain insights into the dynamics of stock price movements.


## Features

  -  Download real-time stock data using Yahoo Finance with the yfinance library.
    
  -  Classify stock price movements into Up, Down, and Stable states.
    
  -  Construct a Markov Chain for modeling state transitions and predicting future states.
    
  -  Train a Logistic Regression model to predict stock price movements using daily returns.
    
  -  Evaluate and compare the performance of Markov Chains and Machine Learning predictions.

## How to Use

  1- Run **fetch_data.py** to download stock data.
    
  2- Use **markov_model.py** to analyze states and build the transition matrix.
    
  3- Train a machine learning model using **ml_model.py**.
    
  4- Visualize results using **visualize.py**.
