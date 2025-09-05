import yfinance as yf
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.data_download import download_data


"""
Normalizes features using rolling z-score. The z-score is computed using the mean
and standard deviation over the number previous intervals given by window_len
"""
def zscore_norm(df: pd.DataFrame, features: list[str], window_size: int = 96) -> pd.DataFrame:
  norm_df = df.copy()
  norm_df.sort_values(["ticker", "Datetime"], inplace=True)
  grouped_df = df.groupby("ticker", group_keys=False)

  for f in features:
    rolling_mean = grouped_df[f].transform(lambda x: x.rolling(window_size, min_periods=window_size).mean())
    rolling_std = grouped_df[f].transform(lambda x: x.rolling(window_size, min_periods=window_size).std(ddof=0))
    #A very small constant is added to the std to prevent division by zero errors
    norm_df[f] = (norm_df[f] - rolling_mean) / (rolling_std + 1e-9)
    norm_df[f] = norm_df[f].shift(1)
  
  return norm_df.dropna()

"""
Method to combine the dataframes from multiple tickers into the complete dataset.
"""
def build_dataset(tickers: list[str], features: list[str]) -> pd.DataFrame:
  #Download ticker data in parallel instead of sequentially to reduce wait time
  ticker_dfs = []
  with ThreadPoolExecutor(max_workers=3) as exe:
    futures = {exe.submit(download_data, t): t for t in tickers}
    for f in as_completed(futures):
      ticker_dfs.append(f.result())
  
  #Combine the list of ticker DataFrames into one DataFrame
  complete_df = pd.concat(ticker_dfs, axis=0, ignore_index=False)
  #Maintain chronological order by sorting by datetime index
  complete_df.sort_index(inplace=True)
  #Replace the datetime index with a numerical index
  complete_df.reset_index(inplace=True)

  #Normalize input features
  complete_df = zscore_norm(complete_df, features)

  return complete_df

"""
Method to split the DataFrame into training and testing DataFrames chronologically per ticker.
"""
def split_df(df: pd.DataFrame, train_split: float, val_split: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  train_list, test_list, val_list = [], [], []

  #Group the Dataframe by ticker and loop through each group
  for _, group in df.groupby("ticker"):
    n = len(group)
    #train split
    i_train = int(n * train_split)
    i_val = int(n * val_split)
    """
    Default split: 70% train, 15% validation, 15% test
    """
    train_list.append(group.iloc[:i_train])
    val_list.append(group.iloc[i_train:i_train+i_val])
    test_list.append(group.iloc[i_train+i_val:])
  
  #Combine the lists of Series into DataFrames
  train_df = pd.concat(train_list)
  val_df = pd.concat(val_list)
  test_df = pd.concat(test_list)

  return train_df, val_df, test_df

"""
Method to create time sequences from chronologically sorted train and test DataFrames
"""
def create_sequence(df: pd.DataFrame, features: list[str], label: str = "return_label", window_size: int = 96) -> tuple[np.ndarray,np.ndarray]:
  X, y = [], []

  #Group the DataFrame by ticker and loop through the groups
  for _, group in df.groupby("ticker"):
    #Get row values for input features for each ticker
    data = group[features].values
    #Get row values for label for each ticker
    labels = group[label].values
    #Add the feature and label values in a sliding window to X and y respectively
    for i in range(len(data) - window_size):
      X.append(data[i:i+window_size])
      y.append(labels[i+window_size])
  
  return np.array(X), np.array(y)

"""
Method to get training, validation and testing data for a list of tickers and features with a given split percentage
"""
def get_train_test_val(tickers: list[str] = None, 
                   features: list[str] = None, 
                   train_split: int = 0.7, 
                   val_split = 0.15) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
  #Default tickers and input features
  if tickers is None:
    tickers = ["SPY", "QQQ", "DIA", "IWM", "VTI"]
  if features is None:
    features = ["Open", "High", "Low", "Close", "Volume", "rsi", "ema", "atr_pct"]

  ticker_df = build_dataset(tickers, features) #Get the DataFrame for all given tickers
  train_df, val_df, test_df = split_df(ticker_df, train_split, val_split) #Split the DataFrame into training and testing portions
  X_train, y_train = create_sequence(train_df, features) #Create training time sequences
  X_val, y_val = create_sequence(val_df, features) #Create validation time sequences
  X_test, y_test = create_sequence(test_df, features) #Create testing time sequences
  return X_train, X_val, X_test, y_train, y_val, y_test