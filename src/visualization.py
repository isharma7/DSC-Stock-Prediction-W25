import pandas as pd
import matplotlib.pyplot as plt
import os

def fix_csv(input, output):
    df = pd.read_csv(input, skiprows = 2) # skip first two rows (messed up)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"] # set columns explicitly
    df["Date"] = pd.to_datetime(df["Date"]) # convert date column to datetime
    df.set_index("Date", inplace = True) # set date as an index
    df.to_csv(output) # new csv

# only visualizes AAPL closing price for now!
def closing_price_visualization(ticker = "AAPL", path="data/"):
    file_path = os.path.join(path, f"{ticker}_clean.csv")

    # read csv
    df = pd.read_csv(file_path, parse_dates=["Date"]) 
    df.set_index("Date", inplace=True)

    # plotting
    plt.plot(df["Close"], label=f"{ticker} Close Price")
    plt.title(f"{ticker} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

if __name__ == "__main__":
    fix_csv("../data/AAPL.csv", "../data/AAPL_clean.csv")
    closing_price_visualization()