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


def simple_regression(input_file, predictor, time):
    df = pd.read_csv(input_file, index_col="Date", parse_dates=True)
    df["Predictor1"] = df[str(predictor)].rolling(int(time)).mean()

    df.dropna(subset=["Predictor1", "Close"], inplace=True)
    X = df[["Predictor1"]]
    Y = df["Close"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=67)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    MAE = mean_absolute_error(Y_test, predictions)
    MSE = mean_squared_error(Y_test, predictions)
    print(f"Predictor: '{predictor}' For Last {time} Days")
    print(f"MAE: {MAE}")
    print(f"MSE: {MSE}\n")


if __name__ == "__main__":
    fix_csv("../data/AAPL.csv", "../data/AAPL_clean.csv")
    simple_regression("../data/AAPL_clean.csv", "Close", 1) # very good
    simple_regression("../data/AAPL_clean.csv", "Close", 7) # good
    simple_regression("../data/AAPL_clean.csv", "Close", 30) # not great


