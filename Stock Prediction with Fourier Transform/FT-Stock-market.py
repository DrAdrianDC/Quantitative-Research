
# Stock prediction using Fourier Transform

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download data
spy500 = yf.download("SPY", start="2010-01-01", end="2024-12-01")

# Error Handling for empty data
if spy500.empty:
    print("Failed to download data. Please check the ticker or internet connection.")
    exit()

# Handle MultiIndex: Extract 'Close' for 'SPY'
if ('Close', 'SPY') not in spy500.columns:
    print("Error: ('Close', 'SPY') column not found in the data.")
    exit()

# Extract and preprocess 'Close' prices
dataset_spy500 = spy500[('Close', 'SPY')].copy()
dataset_spy500.name = 'Close'  # Rename for clarity
dataset_spy500 = dataset_spy500.dropna()

# Add dates as the index
dataset_spy500 = dataset_spy500.reset_index()
dataset_spy500['Date'] = pd.to_datetime(dataset_spy500['Date'])  # Ensure proper datetime format
dataset_spy500.set_index('Date', inplace=True)

# Calculate the Fourier Transform on the 'Close' prices
close_fft = np.fft.fft(dataset_spy500['Close'].tolist())

# Create a DataFrame for Fourier components
fft_df = pd.DataFrame({'fft': close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

# Plot the Fourier Transforms with dates as x-axis
plt.figure(figsize=(14, 7), dpi=100)
plt.plot(dataset_spy500.index, dataset_spy500['Close'], label='Real')  # Plot original data with dates
for num_ in [3, 6, 9]:
    fft_list_m10 = np.copy(close_fft)
    fft_list_m10[num_:-num_] = 0
    transformed = np.fft.ifft(fft_list_m10)
    plt.plot(dataset_spy500.index, transformed.real, label=f'Fourier transform with {num_} components')
plt.xlabel('Date')
plt.ylabel('USD')
plt.title('S&P 500 (Close) stock prices & Fourier transforms')
plt.legend()
#plt.grid(True)
plt.savefig("ft_s&p500_stock_prices.png", dpi=300, bbox_inches='tight')
plt.show()
