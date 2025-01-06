## Stock Prediction with Fourier Transform



Fourier transforms is a mathematical technique that decomposes a signal in the time domain into its underlying frequencies. By analyzing the frequency components of a signal, we can identify patterns and trends that may be difficult to see in the original data.


To apply Fourier transforms to our financial data, we will first calculate the Fourier transform of the closing prices using the NumPy library. This can help us to identify trends, cycles, and other patterns in the data that may not be apparent by simply looking at the raw stock prices. We will then plot the Fourier transforms alongside the actual closing prices to visualize the frequency components of the data.

## Code 

This Python script analyzes the S&P 500 ETF (SPY) historical stock price data using Fourier Transform techniques to identify patterns and trends. Here's what it does:

  -  **Data Download:** Retrieves SPY data from Yahoo Finance between 2010-01-01 and 2024-12-01.
  -  **Error Handling:** Checks if the data is empty or lacks the required 'Close' prices column, ensuring robustness.
  -  **Data Preprocessing:** Extracts, cleans, and formats the 'Close' price data with the Date as the index.
  -  **Fourier Transform:** Applies the Fast Fourier Transform (FFT) to the 'Close' prices, capturing frequency components.
  -  **Visualization:** Reconstructs and visualizes the stock price using various numbers of Fourier components (e.g., 3, 6, 9) to highlight the impact of frequency truncation.
  -  **Output:** Saves the plot as a PNG file and displays it, showing both the original stock prices and Fourier-reconstructed trends.

This code is ideal for exploring the cyclical nature of stock prices and learning signal processing techniques in financial analysis.

## Output

![ft_s p500_stock_prices](https://github.com/user-attachments/assets/6a17d926-a3b5-4c77-ab7a-65e19bb10d10)




The Fourier transform with 3 components appears to capture the general trend of the closing prices, while the Fourier transforms with 6 and 9 components capture additional high-frequency components. In other words, it extracts long and short-term trends. The transformation with 9 components serves as the long-term trend.


## Requirements

- **Python 3.8.3**
  
Required Libraries: The following libraries need to be installed:

**yfinance**: For accessing financial data.

**pandas**: For data manipulation and analysis.

**numpy**: For numerical computations.

**matplotlib**: For creating visualizations.


## Installation

```bash
pip install yfinance pandas numpy matplotlib
```

## Usage
Run the script: 

```bash
python FT-Stock-market.py
```

