# Import necessary libraries:
# - streamlit (for the app)
# - pandas (for loading and handling data)
# - matplotlib.pyplot (for plotting)
# - ARIMA from statsmodels (for time series forecasting)
# - LinearRegression and train_test_split from sklearn (for basic regression)

# Set up the Streamlit app:
# - Set the page title and layout
# - Add a title and short description at the top of the app

# Create dropdown menus:
# - One for choosing the stock (e.g. AAPL, TSLA, SPY)
# - One for choosing the model type ("ARIMA" or "Linear Regression")

# Load the selected stock’s CSV file from the ../data/ folder:
# - Make sure to strip column names
# - Convert the "Date" column to datetime and set it as the index
# - Ensure the "Close" column is numeric and drop rows with missing Close values

# If the user selects ARIMA:
# - Create and fit an ARIMA model on the "Close" column
# - Forecast the next 5 days
# - Plot the actual prices and overlay the forecast as a dashed line
# - Display the forecasted values in the app

# If the user selects Linear Regression:
# - Create a new column called "Day" that numbers each row (e.g., 0, 1, 2, ...)
# - Use the last 60 days of data to train a Linear Regression model
# - Predict the next 5 days using future day numbers
# - Plot the actual prices and overlay the linear forecast
# - Print out the 5 predicted price values in the app





#(Optional Bonus Tasks):
# - Let users change the number of forecast days with a slider
# - Display model evaluation metrics (e.g., MAE, MSE)
# - Allow user to upload their own CSV file
# - Organize sections with st.columns() or st.expander() for better layout
