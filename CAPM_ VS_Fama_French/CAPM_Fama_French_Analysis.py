import yfinance as yf
import pandas as pd
import pandas_datareader.data as reader
import statsmodels.api as sm
import seaborn as sns
import datetime as dt
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

start = dt.datetime(2000, 1, 1)
end = dt.datetime(2026, 1, 1)

Tickers = ["AAPL","XOM", "JPM", "CAT","ORCL","AMZN","KO","^GSPC"]
df = yf.download(Tickers,start=start,end=end,auto_adjust=True)['Close']

# resample daily prices to month-end prices
monthly_returns  = df.resample("ME").last()
monthly_returns  = monthly_returns .pct_change().dropna()

# download Fama-French  factors and risk free rate
factors = reader.DataReader("F-F_Research_Data_5_Factors_2x3","famafrench",dt.datetime(2000,2,1),end)[0]
factors = factors / 100

factors.index = factors.index.to_timestamp("M")

# merge Apple returns with Fama-French factors
merge_ff = monthly_returns[["AAPL"]].merge(factors,left_index=True,right_index=True,how="inner")

# calculate excess returns and add new coloum
merge_ff["AAPL-rf"] = merge_ff["AAPL"] - merge_ff["RF"]

# CAPM regression
y = merge_ff["AAPL-rf"]
X_capm = merge_ff["Mkt-RF"]
X_capm = sm.add_constant(X_capm)

# CAPM regression
capm_model  = sm.OLS(y, X_capm)
capm_results = capm_model .fit()

print("\nCAPM Regression:\n")
print(capm_results.summary())

# 3 factors Fama French

# independent variables
X_ff3  = merge_ff[["Mkt-RF", "SMB", "HML"]]
X_ff3  = sm.add_constant(X_ff3)

# run Fama-French 3-factor regression
ff3_model  = sm.OLS(y, X_ff3)
ff3_results = ff3_model.fit()

print("\nFama-French three-Factors Regreesion:\n")
print(ff3_results.summary())

# 5 factors Fama French

X_ff5  = merge_ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
X_sm = sm.add_constant(X_ff5)

ff5_model = sm.OLS(y, X_sm)
ff5_results = ff5_model.fit()

print("\nFama-French Five-Factors Regression:\n")
print(ff5_results.summary())

# create comparison table

comparison_table = pd.DataFrame({
    "CAPM": [capm_results.params["const"],capm_results.params["Mkt-RF"],
             capm_results.tvalues["Mkt-RF"],capm_results.rsquared,
             capm_results.rsquared_adj,capm_results.aic,capm_results.bic],
    
    "Fama-French 3": [ff3_results.params["const"],ff3_results.params["Mkt-RF"],
        ff3_results.tvalues["Mkt-RF"],ff3_results.rsquared,
        ff3_results.rsquared_adj,ff3_results.aic,ff3_results.bic],
    
    "Fama-French 5": [ff5_results.params["const"],ff5_results.params["Mkt-RF"],
        ff5_results.tvalues["Mkt-RF"],ff5_results.rsquared,
        ff5_results.rsquared_adj,ff5_results.aic,ff5_results.bic]},

index=["Alpha","Market Excess Return Beta","Market Excess Return Beta (t-stat)","R-squared",
    "Adjusted R-squared","AIC","BIC"])

print("\nModels Comparison: CAPM vs Fama-French Models\n")
print(comparison_table)

# SML calculation using S&P 500

rf_annual = merge_ff["RF"].mean()*12
market_return_annual = monthly_returns["^GSPC"].mean()*12

beta_range = np.linspace(0,3,100)
sml_line = (rf_annual + beta_range*(market_return_annual-rf_annual))

plt.figure(figsize=(10,6))
plt.plot(beta_range,sml_line,color="red",label="Security Market Line")
plt.xlim(left=0)
plt.xlabel("Beta")
plt.ylabel("Expected Annual Return")
plt.title("Security Market Line (SML)")
plt.legend()
plt.show()

# CAPM plot

sns.regplot(x="Mkt-RF",y="AAPL-rf",data=merge_ff,line_kws={"color": "red"})
plt.xlabel("Market Excess Return")
plt.ylabel("Apple Excess Return")
plt.title("CAPM Regression")
plt.show()

# Fama-French factors plot

ff5_factors = ["SMB","HML","RMW","CMA"]

fig, axes = plt.subplots(2,2,figsize=(12, 10))
axes = axes.flatten()

for ax, factor in zip(axes, ff5_factors):

    sns.regplot(x=factor,y="AAPL-rf",data=merge_ff,ax=ax,line_kws={"color": "red"})

    ax.set_xlabel(factor)
    ax.set_ylabel("Apple Excess Return")
    ax.set_title(f"{factor} Coefficient")

plt.tight_layout()
plt.show()

# rolling beta (CAPM)

rolling_capm = RollingOLS.from_formula(
    formula="Q('AAPL-rf') ~ Q('Mkt-RF')",data=merge_ff,
    window=36)

capm_rolling_results = rolling_capm.fit()
capm_beta = capm_rolling_results.params["Q('Mkt-RF')"].dropna()

# rolling beta (Fama-French 3 Factor)

rolling_ff3 = RollingOLS.from_formula(
    formula="Q('AAPL-rf') ~ Q('Mkt-RF') + SMB + HML",
    data=merge_ff,window=36)

ff3_rolling_results = rolling_ff3.fit()
ff3_beta = ff3_rolling_results.params["Q('Mkt-RF')"].dropna()

# rolling beta (Fama-French 5 Factor)

rolling_ff5 = RollingOLS.from_formula(
    formula="Q('AAPL-rf') ~ Q('Mkt-RF') + SMB + HML + RMW + CMA",
    data=merge_ff,window=36)

ff5_rolling_results = rolling_ff5.fit()
ff5_beta = ff5_rolling_results.params["Q('Mkt-RF')"].dropna()

# plot

plt.figure(figsize=(12,6))
plt.plot(capm_beta.index, capm_beta, label="CAPM")
plt.plot(ff3_beta.index, ff3_beta, label="Fama-French 3 Factor")
plt.plot(ff5_beta.index, ff5_beta, label="Fama-French 5 Factor")

plt.xlabel("Year")
plt.ylabel("Rolling Market Excess Return Beta ")
plt.title("36-Month Rolling Market Excess Return Beta Comparison ")
plt.legend()
plt.margins(x=0)
plt.show()

# Fama-Macbeth

beta_results = {}
ffm = monthly_returns[Tickers[:-1]].merge(factors[["RF"]],left_index=True,right_index=True,
    how="inner")

# calculate excess returns for every stock
excess_returns = ffm[Tickers[:-1]].subtract(ffm["RF"],axis=0)

for  stock in Tickers[:-1]:

    data = pd.concat([excess_returns[stock],
        factors[["Mkt-RF","SMB","HML","RMW","CMA"]]],axis=1).dropna()

    data.columns = ["excess_return","Mkt-RF","SMB","HML","RMW","CMA"]

    X = sm.add_constant(data[["Mkt-RF","SMB","HML","RMW","CMA"]])

    # run time-series regression:
    model = sm.OLS(data["excess_return"],X).fit()

    beta_results[stock] = model.params

betas = pd.DataFrame(beta_results).T

fm_data = []

# run cross-sectional regressions every month
for date in excess_returns.index:

    for stock in Tickers[:-1]:

        fm_data.append([date,stock,excess_returns.loc[date,stock],
                betas.loc[stock,"Mkt-RF"],betas.loc[stock,"SMB"],
                betas.loc[stock,"HML"],betas.loc[stock,"RMW"],
                betas.loc[stock,"CMA"]])

fm_data = pd.DataFrame(fm_data,
    columns=["month","stock","excess_return","beta","SMB","HML","RMW","CMA"])

# estimate monthly factor risk premiums (gamma values)
monthly_gamma = (fm_data
    .groupby("month")
    .apply(lambda x:smf.ols("""excess_return  ~ beta + SMB + HML + RMW + CMA""",data=x).fit().params))

# calculate the average risk premium over all months and t-statstics
fama_macbeth_results = pd.DataFrame({"Risk Premium":monthly_gamma.mean(),"t-stat":
    monthly_gamma.mean()/(monthly_gamma.std()/np.sqrt(len(monthly_gamma)))})

print("\nFama-Macbeth Results:\n")
print(fama_macbeth_results)
