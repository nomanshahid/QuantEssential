import pandas as pd
import matplotlib.pyplot as plt

def plot_data(df, title="Stock prices"):
	ax = df.plot(title=title)
	ax.set_xlabel("Date")
	ax.set_ylabel("Price")
	plt.show()

def main():
	dfAAPL = pd.read_csv("data/AAPL.csv",index_col="Date",parse_dates=True, usecols=['Date', 'Adj Close'])
	plot_data(dfAAPL, title="Apple Stock Prices 10-Year Chart")


if __name__ == "__main__":
	main()


# Load Data
# Clean Data (make data based on excel video columns)
# Calculate RSI
# Plot RSI 
# Calculate MACD
# Plot MACD

#Build tensorflow model based on these params
