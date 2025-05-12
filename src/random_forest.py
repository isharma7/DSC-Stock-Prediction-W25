# random forest.py 
#moving average and price change

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

#load csv
df = pd.read_csv("data/SPY.csv", skiprows=2)
#df = pd.read_csv("data/AAPL.csv")
df.dropna(inplace = True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Close'] = df['Unnamed: 1']
df['High'] = df['Unnamed: 2']
df['Low'] = df['Unnamed: 3']
df['Open'] = df['Unnamed: 4']
df['Volume'] = df['Unnamed: 5']

#create features
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()
df['Close_lag'] = df['Close'].shift(1)
df['Volume_lag'] = df['Volume'].shift(1)

df.dropna(inplace=True)

#parameters and target
x = df[['Open']] 
y = df['Close'] #Close

#test vs training data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)

# Create a linear regression model
model = RandomForestRegressor(n_estimators = 100, random_state = 42)
model.fit(x_train, y_train)

# predict model
y_predict = model.predict(x_test) 

#plot
# Evaluate
mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R^2 Score: {r2}')


# Plot actual vs predicted close prices
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_predict, label='Predicted')
plt.title('Actual vs Predicted Closing Prices')
plt.xlabel('Sample')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

