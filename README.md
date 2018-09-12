# QuantEssential

Deep learning models built with TensorFlow to predict equity price movements using RSI/MACD flow indicators.

## Purpose Of This Project
The goal here was to dive deeper into understanding the role of machine learning in the world of trading. The stock market is a complex environment and we must understand that AI/ML will only solve the problem to a certain degree. The financial models and algorithms explored here are simply for exploratory purposes using historical data in a test environment. Future plans are to hook it up to place real orders since order entry is no longer available for personal applications through the [Questrade API](https://www.questrade.com/api).

### The Financial Side: Developing a Theoretical Model

The financial model was built upon the understanding that stocks going through a consolidation period can have price movement predicted by two key flow indicators: **RSI** and **MACD**. 

The [RSI](https://www.investopedia.com/terms/r/rsi.asp) (Relative Strength Index) is a momentum indicator that helps measure if a stock is being traded above or below its true value. The formula is as follows:
<p align="center">
<img src="https://user-images.githubusercontent.com/15072395/44473744-042b7300-a5ff-11e8-9a07-1a89ffb20794.PNG" width="350" />
</p>

A value ≥70 indicates a stock is overvalued and ≤30 indicates it is undervalued, our analysis concluded that an RSI value in 50's is the tipping point for the price movement, so anything above 50 indicates an uptrend and vice versa.

We also use the [MACD](https://www.investopedia.com/terms/m/macd.asp) (Moving Average Convergence Divergence) momentum indicator to identify buy opportunities. MACD is a technical indicator that anticipates future movements based on the difference in short-term and long-term price trends. We also plot a signal line and determine whenever MACD crosses above the signal line it is a bullish signal, and a bearish signal when it falls below. The formulas are as follows: 
<p align="center">
<img src="https://user-images.githubusercontent.com/15072395/44473691-e231f080-a5fe-11e8-8bf6-c4cd6fca062f.PNG" width="350" />
</p>

Note that [EMA](https://www.investopedia.com/terms/e/ema.asp) stands for the *Exponential Moving Average*. 

Our model uses these two technical indicators by first identifying trend movement and oversold/overbought stocks with the RSI indicator, then confirming buy/sell signals with MACD.

### The Technical Side: Building a Recurrent Neural Network 
Having defined the financial model, it was clear that our data set would be trained on 3 key inputs: Adjusted Close, RSI, and MACD. Since we are interested in our data being persistent as the model is trained, we chose to use a Recurrent Neural Net along with LSTM cells to allow the network to remember long-term dependencies.

// Need to complete

## How To Use It

Currently, only Apple's stock ($AAPL) is being fed into the program with daily historical data from the past 10 years. The analysis is done for daily data from 2017 specifically. In order to view the training results, predictions, charts and model accuracy simply run the main program with the following command (after ensuring all dependencies are installed):

    $ python quantessential.py
An important next step under development is to allow the program to take in stock symbols as a parameter and fetch the data automatically from the Questrade API.

## Built With

* [TensorFlow](https://www.tensorflow.org/) - For developing the deep learning models
* [Questrade API](https://www.questrade.com/api) - For retrieving market data for testing (*under development!*)
* NumPy, pandas and lots of other useful Python libraries :)

## Acknowledgments

Special thanks to Arjun Nypaul for providing his insight to develop the financial model. 
