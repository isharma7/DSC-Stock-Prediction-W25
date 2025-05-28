from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def fix_csv(input_file, output_file):
    df = pd.read_csv(input_file, skiprows = 2) # skip first two rows (messed up)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"] # set columns explicitly
    df["Date"] = pd.to_datetime(df["Date"]) # convert date column to datetime
    df.set_index("Date", inplace = True) # set date as an index
    df.to_csv(output_file) # new csv


def random_forest(input_file):
    df = pd.read_csv(input_file, index_col="Date", parse_dates=True)
    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_30"] = df["Close"].rolling(30).mean()

    df.dropna(subset=["MA_7", "MA_30", "Close"], inplace=True)
    X = df[["MA_7", "MA_30"]]
    Y = df["Close"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=67)
    model = RandomForestRegressor(n_estimators=1000, random_state=67)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    MAE = mean_absolute_error(Y_test, predictions)
    MSE = mean_squared_error(Y_test, predictions)
    R2 = r2_score(Y_test, predictions)
    print(f"MAE: {MAE}")
    print(f"MSE: {MSE}")
    print(f"R2: {R2}\n")


if __name__ == "__main__":
    fix_csv("../data/AAPL.csv", "../data/AAPL_clean.csv")
    random_forest("../data/AAPL_clean.csv")


