import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load monthly sales data
df = pd.read_csv("data/monthly_sales.csv")

# Create numeric month index
df["Month_num"] = np.arange(len(df))

X = df[["Month_num"]]
y = df["Sales"]

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Forecast next 6 months
future_months = np.arange(len(df), len(df) + 6)
future_df = pd.DataFrame({"Month_num": future_months})

forecast = model.predict(future_df)

forecast_df = pd.DataFrame({
    "Month": future_months,
    "Forecasted_Sales": forecast
})

print(forecast_df)
