import pandas as pd
import matplotlib.pyplot as plt

def calculate_rsi(df):
	diff = df['Adj Close'].diff()
	up, down = diff.copy(), diff.copy()
	up[up < 0] = 0
	down[down > 0] = 0
	df['Upward Movement'] = up
	df['Downward Movement'] = abs(down)
	df['Avg. 14-Day Up Closes'] = df['Upward Movement'].rolling(14).mean() # using 14 day periods
	df['Avg. 14-Day Down Closes'] = df['Downward Movement'].rolling(14).mean() # using 14 day periods
	df['Relative Strength'] = df['Avg. 14-Day Up Closes'] / df['Avg. 14-Day Down Closes']
	df['RSI'] = 100 - (100/(1+df['Relative Strength']))

def plot_rsi(df, title="RSI"):
	ax = df['RSI'].plot(title=title)
	ax.set_xlabel("Date")
	ax.set_ylabel("RSI")
	plt.axhline(y=30,color='k')
	plt.axhline(y=70,color='k')
	plt.show()

def plot_data(df, title="Stock prices"):
	ax = df['Adj Close'].plot(title=title)
	ax.set_xlabel("Date")
	ax.set_ylabel("Price")
	plt.show()

def main():
	dates = pd.date_range('2017-01-01', '2017-12-31') # only 2017 data testing for now
	df = pd.DataFrame(index=dates)
	dfAAPL = pd.read_csv("data/AAPL.csv",index_col="Date",parse_dates=True, usecols=['Date', 'Adj Close'])
	df = df.join(dfAAPL)
	df = df.dropna()
	print(df.head(20))
	calculate_rsi(df)
	plot_data(df, title="Apple Stock Prices 2017 Chart")
	plot_rsi(df)


if __name__ == "__main__":
	main()


# Load Data
# Clean Data (make data based on excel video columns)
# Calculate RSI
# Plot RSI 
# Calculate MACD
# Plot MACD

#Build tensorflow model based on these params
