import pandas as pd
import numpy as np
from datetime import datetime

stock = pd.read_csv("sphist.csv")

stock["Date"] = pd.to_datetime(stock["Date"])
df = stock[stock["Date"] < datetime(year = 2015, month = 4, day = 1)]
df = df.sort_values(by = "Date",ascending=True)
df = df.drop(["Open", "High", "Low","Volume"], axis =1)

df["day_5"] = pd.rolling_mean(df["Close"], 5, min_periods=5)
df["day_5"] = df["day_5"].shift(periods = 1, axis = 0)
df["day_30"] = pd.rolling_mean(df["Close"], 30, min_periods=30)
df["day_30"] = df["day_30"].shift(periods = 1, axis = 0)
df["day_365"] = pd.rolling_mean(df["Close"], 365, min_periods=365)
df["day_365"] = df["day_365"].shift(periods = 1, axis = 0)

df = df[df["Date"]>datetime(year=1951, month=1, day=3)]
df = df.dropna(axis=0)
test = df[df["Date"]<datetime(year = 2013, month = 1, day = 1)]
train = df[df["Date"]>=datetime(year=2013, month=1, day=1)]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
lr = LinearRegression()
lr.fit(train[["day_5","day_30","day_365"]], train["Close"])
predictions = lr.predict(test[["day_5", "day_30", "day_365"]])
mae = mean_absolute_error(test["Close"], predictions)
print(mae)
