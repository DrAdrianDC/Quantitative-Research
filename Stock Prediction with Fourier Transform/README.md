## Stock Prediction with Fourier Transform


In this project the main focus is apply Fourier Transforms for a predictive model to analyze and forecast time series data. 

Fourier transforms a mathematical technique that decomposes a signal in the time domain into its underlying frequencies. By analyzing the frequency components of a signal, we can identify patterns and trends that may be difficult to see in the original data.


To apply Fourier transforms to our financial data, we will first calculate the Fourier transform of the closing prices using the NumPy library. We will then plot the Fourier transforms alongside the actual closing prices to visualize the frequency components of the data.



![ft_s p500_stock_prices](https://github.com/user-attachments/assets/6a17d926-a3b5-4c77-ab7a-65e19bb10d10)




The Fourier transform with 3 components appears to capture the general trend of the closing prices, while the Fourier transforms with 6 and 9 components capture additional high-frequency components. In other words, it extracts long and short-term trends. The transformation with 9 components serves as the long-term trend.


## Requirements

- Python 3.8.3
- NumPy library
- yfinance library

## Usage
Run the script: 

```bash
python FT-Stock-market.py
```

