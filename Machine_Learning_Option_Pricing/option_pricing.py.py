import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import numpy as np
from scipy.stats import norm
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from pandas_datareader import data as web
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# black-scholes function
def black_scholes_price(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# download data for NASDAQ 100
url = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"

tickers = pd.read_html(
    StringIO(requests.get(url, headers={"User-Agent":"Mozilla/5.0"}).text)
)[0]["Ticker"].tolist()

prices = yf.download(tickers,start="2020-01-01",end="2026-01-01",auto_adjust=True)["Close"]

# remove companies with incomplete data
prices = prices.dropna(axis=1, how="any")

# calculate returns
prices = prices.resample("ME").last()
returns = prices.pct_change()

# one year lockback rolling volatility
volatility = returns.rolling(12).std() * np.sqrt(12)

# Strike prices are defined as 90%, 100%, and 110% of the underlying stock price
# to represent in-the-money, at-the-money, and out-of-the-money call options.
# Option maturities of 1 month, 6 months, and 1 year are considered.

strike_levels = [0.90, 1.00, 1.10]
maturities = [30/365,180/365,365/365]

# download monthly kenneth french risk-free rate
ff = web.DataReader("F-F_Research_Data_Factors","famafrench",start="2021-01-01",end="2026-01-01")

rf = ff[0][["RF"]].copy()
rf["RF"] = rf["RF"] / 100

# annualize the daily risk-free rate
rf["RF"] = (1 + rf["RF"])**12 - 1
rf.index = rf.index.to_timestamp("M")

# 2020 data is used as a lookback period to calculate 12-month historical volatility.
# The Black-Scholes dataset starts from 2021 after volatility estimation is completed.

volatility = volatility[volatility.index >= "2021-01-01"]
prices = prices[prices.index >= "2021-01-01"]

# computes option prices
rows = []
for ticker in prices.columns:

    stock = prices[ticker]
    sigma = volatility[ticker]

    for date in stock.index:
        S = stock.loc[date]
        vol = sigma.loc[date]
        r = rf.loc[date, "RF"]

        for level in strike_levels:
            K = S * level

            for T in maturities:
                option_price = black_scholes_price(S=S,K=K,r=r,T=T,sigma=vol)
                rows.append({"Date": date,"Ticker": ticker,"S": S,"K": K,"r": r,"T": T,"sigma": vol,
                    "BlackScholesPrice": option_price})

option_data = pd.DataFrame(rows)

# splitting and standardizing data

train_data = option_data[option_data["Date"] < "2025-01-01"]
test_data = option_data[option_data["Date"] >= "2025-01-01"]

features = ["S", "K", "r", "T", "sigma"]

preprocessor = ColumnTransformer(transformers=[("scale",StandardScaler(),features)],remainder="drop")

# support vector regression (SVR)

svr_pipeline = Pipeline([
    ("preprocessor", preprocessor),("regressor", SVR(kernel="rbf"))])

svr_pipeline.fit(
    train_data[features],train_data["BlackScholesPrice"])

# random forest

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor",RandomForestRegressor(
            n_estimators=20,
            min_samples_leaf=500,
            random_state=42))])

rf_pipeline.fit(
    train_data[features],train_data["BlackScholesPrice"])

# deep neural network

deep_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor",MLPRegressor(
            hidden_layer_sizes=(10,10,10),
            activation="logistic",
            solver="adam",
            max_iter=1000,
            random_state=42))])

deep_pipeline.fit(train_data[features],train_data["BlackScholesPrice"])

# xgboost

xgb_pipeline = Pipeline([
    ("regressor",
     XGBRegressor(
         n_estimators=300,
         max_depth=4,
         random_state=42))])

xgb_pipeline.fit(train_data[features],train_data["BlackScholesPrice"])

# predicting the results 

test_X = test_data[features]
predictive_performance = pd.concat(
    [
        test_data.reset_index(drop=True),
        pd.DataFrame({
            "Random Forest": rf_pipeline.predict(test_X),
            "SVR": svr_pipeline.predict(test_X),
            "Deep NN": deep_pipeline.predict(test_X),
            "XGBoost": xgb_pipeline.predict(test_X)})],axis=1)

predictive_performance = predictive_performance.melt(
    id_vars=["Date","Ticker","S","K","r","T","sigma","BlackScholesPrice"],
    var_name="Model",
    value_name="Predicted")

# compute errors

predictive_performance["spot_minus_strike"] = (predictive_performance["S"]- predictive_performance["K"])
predictive_performance["pricing_error"] = (predictive_performance["Predicted"]- predictive_performance["BlackScholesPrice"]).abs()

# plot

models = predictive_performance["Model"].unique()

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
axes = axes.flatten()

for ax, model in zip(axes, models):

    subset = predictive_performance[
        predictive_performance["Model"] == model]

    ax.scatter(
        subset["spot_minus_strike"],
        subset["pricing_error"],
        alpha=0.08)

    ax.set_title(model)
    ax.set_xlabel("Spot Price − Strike Price")
    ax.set_ylabel("Absolute Pricing Error")
    ax.grid(True)
plt.tight_layout()
plt.show()

# Model evaluation summary

evaluation_results = []

for model in models:

    pred = predictive_performance[
        predictive_performance["Model"] == model
    ]["Predicted"]

    actual = predictive_performance[
        predictive_performance["Model"] == model
    ]["BlackScholesPrice"]

    pricing_error = predictive_performance[
        predictive_performance["Model"] == model
    ]["pricing_error"]

    evaluation_results.append({
        "Model": model,
        "Mean Pricing Error": pricing_error.mean(),
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(mean_squared_error(actual, pred)),
        "R2": r2_score(actual, pred)})

evaluation_table = pd.DataFrame(evaluation_results)
print(evaluation_table.round(4).to_string(index=False))