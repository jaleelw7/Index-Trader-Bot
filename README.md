# Index Trader Bot

### Overview

[Torch Index](https://www.indextorch.com/) is a fullstack web app that predicts buy, hold, and sell probabilities for a US equity using a custom neural network made in Pytorch. The app also displays the chart for the equity over a selected period of time.

### Data

* Data was collected from 5 ETFs (SPY, QQQ, DIA, IVM, VTI) over 2 years with 1 hour intervals. Training data was retrieved using yfinance.
* OHLCV features were fed into the model along with technical indicators calculated using TA-Lib: RSI, EMA, ATR(as a perecentage of Close price). These features were normalized with z-score normalization.
* Return labels were calculated using an hourly return threshold of 0.5%.

## Model

A Temporal Convolutional Network (TCN) multiclass classifier with 3 residual blocks built using Pytorch. Accepts time series data (sequence length of 96 was used). The default parameters of the TCNModel class are the same as the final model after several adjustments to maximize train/test accuracy. Likewise, the loss function and optimizer parameters in the final version of `train.py` are the same as those used to train the final model. Class weights were implemented as the training dataset was ~70% hold labels. On the test dataset (SPY, QQQ, DIA, IVM, VTI), final model accuracy was found to be ~68%. The state_dict for the final model is stored in the `artifacts/models` directory.

## Backend

A Flask backend hosted on an Amazon EC2 instance. The backend has 3 endpoints:
* candles- Fetches ticker data to display its chart.
* prediction- Creates a time sequence from ticker data and passes it to the trained model. Returns prediction probabilities.
* healthz- Health endpoint.

Data for the backend APIs is retrieved with Tiingo instead of yfinance to avoid IP throttling issues.

## Frontend

A React frontend that allows users to enter a ticker to see its buy/hold/sell prediction probailities and chart over a period of time.
* Hosted on AWS Amplify
* Chart is displayed using Recharts
