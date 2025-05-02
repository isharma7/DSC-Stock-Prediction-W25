import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error  
from sklearn.model_selection import train_test_split

# Load the data
aapl = pd.read_csv('C:\\Users\\pokem\\.vscode\\DSC-Stock-Prediction-W25\\data\\AAPL.csv')


# Preview data


# Drop any metadata rows (e.g., headers within the CSV body)
aapl = aapl[~aapl['Price'].isin(['Ticker', 'Date'])].copy()

# Rename columns
aapl.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Convert columns to numeric (except 'Date')
for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
    aapl[col] = pd.to_numeric(aapl[col], errors='coerce')
aapl['Date'] = pd.to_datetime(aapl['Date'])

# Drop rows with missing data
aapl.dropna(inplace=True)

print(aapl.head())
print(aapl.info())
print(aapl.describe())
# Define features
features = ['Open', 'High', 'Low', 'Close', 'Volume']


# Create the target: next day's adjusted close
aapl['Target'] = aapl['Close'].shift(-1)
aapl.dropna(inplace=True)  # remove last row with NaN target

# Set X and y
X = aapl[features]
y = aapl['Target']
print(X.shape, y.shape)

# Split data (no shuffle to keep time order)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Optional: plot predictions vs true values
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('AAPL Next-Day Price Prediction')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.show()

# Plot prediction errors (residuals)
errors = y_test - y_pred

plt.figure(figsize=(10, 4))
plt.plot(errors)
plt.axhline(0, color='red', linestyle='--')
plt.title("Prediction Errors (Residuals)")
plt.xlabel("Time Step")
plt.ylabel("Error")
plt.show()


