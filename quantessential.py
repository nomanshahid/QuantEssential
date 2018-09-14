import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


#time for some machine learning 
def rnn_adj_close():
	mnist = tf.keras.datasets.mnist

	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0

	model = tf.keras.models.Sequential([
	  tf.keras.layers.Flatten(),
	  tf.keras.layers.Dense(512, activation=tf.nn.relu),
	  tf.keras.layers.Dropout(0.2),
	  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
	])
	model.compile(optimizer='adam',
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=5)
	model.evaluate(x_test, y_test)

def calculate_macd(df):
	df['12-Day EMA'] = df['Adj Close'].ewm(span=12).mean()
	df['26-Day EMA'] = df['Adj Close'].ewm(span=26).mean()
	df['MACD'] = df['12-Day EMA'] - df['26-Day EMA']
	df['Signal Line'] = df['MACD'].ewm(span=9).mean()
	
def plot_macd(df, title="MACD"):
	plt.figure(3)
	ax = df[['MACD', "Signal Line"]].plot(title=title)
	ax.set_xlabel("Date")
	ax.set_ylabel("MACD")

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
	plt.figure(2)
	ax = df['RSI'].plot(title=title, color="green")
	ax.set_xlabel("Date")
	ax.set_ylabel("RSI")
	plt.axhline(y=30,color='k')
	plt.axhline(y=70,color='k')

def plot_data(df, title="Stock prices"):
	plt.figure(1)
	ax = df['Adj Close'].plot(title=title, label='Adj Close')
	ax.set_xlabel("Date")
	ax.set_ylabel("Price")

def main():
	dates = pd.date_range('2017-01-01', '2017-12-31') # only 2017 data testing for now
	df = pd.DataFrame(index=dates)
	dfAAPL = pd.read_csv("data/AAPL.csv",index_col="Date",parse_dates=True, usecols=['Date', 'Adj Close'])
	df = df.join(dfAAPL)
	df = df.dropna()
	calculate_rsi(df)
	calculate_macd(df)
	plot_data(df, title="Apple Stock Prices 2017 Chart")
	plot_rsi(df)
	plot_macd(df)
	plt.show()


if __name__ == "__main__":
	main()

