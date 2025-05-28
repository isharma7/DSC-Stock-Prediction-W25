from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd


def fix_csv(input_file, output_file):
    df = pd.read_csv(input_file, skiprows = 2) # skip first two rows (messed up)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"] # set columns explicitly
    df["Date"] = pd.to_datetime(df["Date"]) # convert date column to datetime
    df.set_index("Date", inplace = True) # set date as an index
    df.to_csv(output_file) # new csv

# multiple linear regression

def multiple_regression1(input_file):
    df = pd.read_csv(input_file, index_col="Date", parse_dates=True)
    df["PCT_Change5"] = df["Close"].pct_change(5)
    df["PCT_Change1"] = df["Close"].pct_change(1)
    df["MA_7"] = df["Close"].rolling(7).mean()

    df.dropna(subset=["PCT_Change5", "PCT_Change1", "MA_7", "Close"], inplace=True)
    X = df[["PCT_Change5", "PCT_Change1", "MA_7"]]
    Y = df["Close"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=67)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    MAE = mean_absolute_error(Y_test, predictions)
    MSE = mean_squared_error(Y_test, predictions)
    print(f"MAE: {MAE}")
    print(f"MSE: {MSE}\n")

def multiple_regression2(input_file):
    df = pd.read_csv(input_file, index_col="Date", parse_dates=True)
    df["Close_Volume"] = df["Close"] / df["Volume"]
    df["Open_Volume"] = df["Open"] / df["Volume"]
  
    df.dropna(subset=["Open_Volume", "Close_Volume", "Close"], inplace=True)
    X = df[["Open_Volume", "Close_Volume"]]
    Y = df["Close"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=67)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    MAE = mean_absolute_error(Y_test, predictions)
    MSE = mean_squared_error(Y_test, predictions)
    print(f"MAE: {MAE}")
    print(f"MSE: {MSE}\n")

def multiple_regression3(input_file):
    df = pd.read_csv(input_file, index_col="Date", parse_dates=True)
    df["MA_Volume_7"] = df["Volume"].rolling(7).mean()
    df["MA_Volume_30"] = df["Volume"].rolling(30).mean()
  
    df.dropna(subset=["MA_Volume_7", "MA_Volume_30", "Close"], inplace=True)
    X = df[["MA_Volume_7", "MA_Volume_30"]]
    Y = df["Close"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=67)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    MAE = mean_absolute_error(Y_test, predictions)
    MSE = mean_squared_error(Y_test, predictions)
    print(f"MAE: {MAE}")
    print(f"MSE: {MSE}\n")

def multiple_regression4(input_file):
    df = pd.read_csv(input_file, index_col="Date", parse_dates=True)
    df["PCT_Change7"] = df["Close"].pct_change(7)
    df["MA_7"] = df["Close"].rolling(7).mean()

    df.dropna(subset=["PCT_Change7", "Close"], inplace=True)
    X = df[["PCT_Change7", "MA_7"]]
    Y = df["Close"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=67)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    MAE = mean_absolute_error(Y_test, predictions)
    MSE = mean_squared_error(Y_test, predictions)
    print(f"MAE: {MAE}")
    print(f"MSE: {MSE}")

if __name__ == "__main__":
    fix_csv("../data/AAPL.csv", "../data/AAPL_clean.csv")
    multiple_regression1("../data/AAPL_clean.csv") # pretty good
    multiple_regression2("../data/AAPL_clean.csv") # terrible
    multiple_regression3("../data/AAPL_clean.csv") # pretty bad
    multiple_regression4("../data/AAPL_clean.csv") # pretty good