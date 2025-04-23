import pandas as pd
import os

def moving_average(filepath, window):
    #read csv
    df = pd.read_csv(filepath, skiprows=2, parse_dates=['Date'])

    #sort by date
    df = df.sort_values('Date')
    
    #calculate 7-day moving average
    df['MA'] = df['Unnamed: 2'].rolling(window).mean()

    return df
df = moving_average('data/AAPL.csv', 10)
print(df[['Date', 'Unnamed: 2', 'MA']].tail())
    
