import streamlit as st

forecasted_price = 20.32
actual_price = 150.00
mae = 2.50
mse = 7.50
r2 = 0.94

st.subheader(" Forecast Summary and Metrics")

delta = forecasted_price - actual_price
st.metric("Predicted Price", f"${forecasted_price:.2f}", delta=f"{delta:.2f}")

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("R^2", f"{r2:.2f}")