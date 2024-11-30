# Stock Price Prediction Using Markov Chains and Machine Learning


## Overview

This project demonstrates the use of **Markov Chains and Machine Learning** for **predicting stock price movements**. By analyzing historical stock data, we classify daily price movements into discrete states (Up, Down, Stable) and model the transitions between these states using a Markov Chain. Additionally, we train a machine learning model to compare its predictive performance against the Markov model.



## Motivation

Predicting stock prices is a challenging task due to the inherent stochastic nature of financial markets. This project explores a hybrid approach, combining probabilistic Markov models with machine learning, to gain insights into the dynamics of stock price movements.

## Why using Markov chains?
Markov chains are useful in finance, for instance, Modeling Stock Prices: Markov chains are often used to model stock prices or financial instruments where future states (e.g., price changes) depend only on the present state and not on past states. 
The idea is to represent different possible price levels or returns as discrete states, and the Markov chain determines the probabilities of transitioning from one state to another.

For example, stock prices can be categorized into states like "up", "down", or "neutral." The Markov chain would calculate the probability of the price going from "up" to "down", or staying in "neutral", given the current state. By using these probabilities, analysts can predict future price movements or simulate possible price scenarios, helping in risk management, option pricing, and decision-making.
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
