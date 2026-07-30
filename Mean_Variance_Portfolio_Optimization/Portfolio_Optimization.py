import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas_datareader.data as web

Tickers = ['AAPL','AMZN','GOOGL','CAT','XOM']

data = yf.download(Tickers,start="2024-01-01",end="2025-12-31",auto_adjust=True)['Close']

# download daily risk free rate
ff = web.DataReader('F-F_Research_Data_Factors_daily','famafrench')[0]
ff = ff[(ff.index >= '2024-01-01') &(ff.index <= '2024-12-31')]
rf = ff['RF'].mean()/100*252

def calc_performance(data,year=2024) :
    data = data[data.index.year == year]
    daily_returns = data.pct_change().dropna()    
    cumul_ret = (1 + daily_returns).prod() - 1
    vol = daily_returns.std()*np.sqrt(252)           # annualized volatility
    ret = daily_returns.mean()* 252                  # annualized mean return
    sharpe_ratio = (ret-rf)/vol

    performance = pd.DataFrame({
        'Return':ret,
        'Volatility':vol,
        'Sharpe Ratio':sharpe_ratio,
        'cumulative ret':cumul_ret})

    return performance, ret, daily_returns

performance, ret, daily_returns = calc_performance(data, year=2024)
print(performance.nlargest(5, 'Return'))

# monte carlo simulation

np.random.seed(42)
num_of_portfolios = 5000
number_of_stocks = len(Tickers)
cov_matrix = daily_returns.cov() * 252

all_weights = np.zeros((num_of_portfolios,number_of_stocks ))
ret_m = np.zeros(num_of_portfolios)
vol_m = np.zeros(num_of_portfolios)
sharpe_m = np.zeros(num_of_portfolios)

# start the simulations.
for i in range(num_of_portfolios):

    weights = np.random.random(number_of_stocks)
    weights = weights / np.sum(weights)
    all_weights[i, :] = weights
    ret_m[i] = np.dot(weights, ret)
    vol_m[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_m[i] = (ret_m[i]-rf)/vol_m[i]

simulations = pd.DataFrame([ret_m, vol_m, sharpe_m, all_weights]).T
simulations.columns = ['Returns','Volatility','Sharpe Ratio','Portfolio Weights']

# return the max sharpe ratio from the run.
max_sharpe_ratio = simulations.loc[simulations['Sharpe Ratio'].idxmax()]

# return the min volatility from the run.
min_volatility = simulations.loc[simulations['Volatility'].idxmin()]

pd.set_option('display.max_colwidth', None)

print('\nMonte Carlo Simulation Results:')
print('\nMax Sharpe Ratio Portofolio:')
print(max_sharpe_ratio.drop('Portfolio Weights'))
print('\nPortfolio Allocation:')
print(pd.DataFrame({'Ticker': Tickers,'Weight': max_sharpe_ratio['Portfolio Weights']}))

print('\nMin Volatility Portofolio:')
print(min_volatility.drop('Portfolio Weights'))
print('\nPortfolio Allocation:')
print(pd.DataFrame({'Ticker': Tickers,'Weight': min_volatility['Portfolio Weights']}))

# markowitz portfolio optimization

def sharpe_ratio(weights, ret, cov_matrix,rf):
    portofolio_return = np.dot(weights, ret)
    portofolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix,weights)))
    sharpe_ratio = (portofolio_return-rf) / portofolio_volatility
    return -sharpe_ratio

constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1 })
bounds = tuple((0,1) for _ in range(len(Tickers)))

intial_weights = np.ones(len(Tickers)) / len(Tickers)

result = minimize(sharpe_ratio,intial_weights,args=(ret,cov_matrix,rf),
    method='SLSQP',bounds=bounds,constraints=constraints)

optimal_weights = result.x
opt_return = np.dot(optimal_weights, ret)
opt_volatility = np.sqrt(np.dot(optimal_weights.T,np.dot(cov_matrix, optimal_weights)))
optimal_sharpe_ratio = (opt_return-rf)/opt_volatility

print('\nMarkowitz Portfolio Optimization Results:')

print("\nOptimal Weights:")
for ticker, weight in zip(Tickers, optimal_weights):
    print(f"{ticker}: {weight:.6f}")

print(f"\nReturns: {opt_return:.8f}")
print(f"Volatility: {opt_volatility:.8f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.8f}")

# efficient frontier

target_returns = np.linspace(simulations['Returns'].min(),simulations['Returns'].max(),100)
bounds = tuple((0,1) for _ in range(len(Tickers)))
initial_weights = np.ones(len(Tickers)) / len(Tickers)

frontier_volatility = []

for target_return in target_returns:

    constraints = [{'type': 'eq','fun': lambda weights: np.sum(weights) - 1},
        {'type': 'eq','fun': lambda weights: np.dot(weights, ret) - target_return}]

    result = minimize(
        lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))),
        initial_weights,method='SLSQP',bounds=bounds,constraints=constraints)

    frontier_volatility.append(result.fun)

# capital market line

cml_volatility_range = np.linspace(0.12,0.22,100)
cml_returns = rf + ((opt_return-rf)/opt_volatility)*cml_volatility_range

# plot

plt.figure(figsize=(10,6))
scatter = plt.scatter(simulations['Volatility'],simulations['Returns'],c=simulations['Sharpe Ratio']
    ,alpha=0.5,s=8)

plt.plot(frontier_volatility,target_returns,linestyle='--',linewidth=2,label='Efficient Frontier')
plt.scatter(opt_volatility,opt_return, marker='o',s=100,color='black',label='Optimal Portfolio')

plt.plot(cml_volatility_range,cml_returns,color='darkred',linewidth=2,label='Capital Market Line')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.title('Mean-Variance Efficient Frontier with Capital Market Line')
plt.legend()
plt.show()

# out-of-sample portofolio performance evaluation

testing_data = data[data.index.year == 2025]

testing_daily_returns = testing_data.pct_change().dropna()

training_portfolio_returns = np.dot(daily_returns, optimal_weights)
testing_portfolio_returns = np.dot(testing_daily_returns, optimal_weights)

training_cumulative_return = np.cumprod(1 + training_portfolio_returns)
testing_cumulative_return = np.cumprod(1 + testing_portfolio_returns)

training_sharpe_ratio = ((training_portfolio_returns.mean() * 252 - rf)/ (training_portfolio_returns.std() * np.sqrt(252)))

# download daily risk free rate for 2025
ff_2025 = web.DataReader('F-F_Research_Data_Factors_daily','famafrench')[0]
ff_2025 = ff_2025[(ff_2025.index >= '2025-01-01') &(ff_2025.index <= '2025-12-31')]
rf_2025 = ff_2025['RF'].mean()/100*252
testing_sharpe_ratio = ((testing_portfolio_returns.mean()*252 - rf_2025) / (testing_portfolio_returns.std()*np.sqrt(252)))

print("\nPerformance Comparison ( Optimized Portfolio 2024 vs 2025 )")
print("\n2024 Performance:")
print(f"Total Return: {(training_cumulative_return[-1]-1)}")
print(f"Sharpe Ratio: {training_sharpe_ratio}")

print("\n2025 Performance:")
print(f"Total Return: {(testing_cumulative_return[-1] - 1)}")
print(f"Sharpe Ratio: {testing_sharpe_ratio}")

# plot

plt.figure(figsize=(10,5))

plt.plot(training_cumulative_return,label="Training 2024")
plt.plot(np.arange(len(testing_cumulative_return)),testing_cumulative_return,label="Testing 2025")
plt.xlabel("Trading Days")
plt.ylabel("Cumulative Returns")
plt.title("Optimal Portofolio 2024 vs 2025 Performance")
plt.legend()
plt.xlim(left=0)   
plt.show()

# download S&P 500 data for testing period
sp500_data = yf.download("^GSPC",start="2025-01-01",end="2025-12-31",auto_adjust=True)['Close'].squeeze()

sp500_daily_returns = sp500_data.pct_change().dropna()
sp500_cumulative_return = np.cumprod(1 + sp500_daily_returns)

sp500_sharpe_ratio = ((sp500_daily_returns.mean() * 252 - rf_2025) /
                      (sp500_daily_returns.std() * np.sqrt(252)))

portfolio_total_return = testing_cumulative_return[-1] - 1
sp500_total_return = sp500_cumulative_return.iloc[-1] - 1

print("\nPerformance Comparison ( Optimized Portfolio 2025 vs S&P 500 )")
print("\nOptimized Portfolio 2025:")
print(f"Total Return: {portfolio_total_return:.2%}")
print(f"Sharpe Ratio: {testing_sharpe_ratio:.3f}")

print("\nS&P 500 Benchmark:")
print(f"Total Return: {sp500_total_return:.2%}")
print(f"Sharpe Ratio: {sp500_sharpe_ratio:.3f}")

# Plot cumulative performance

plt.figure(figsize=(10,5))

plt.plot(np.arange(len(testing_cumulative_return)),testing_cumulative_return,label="Optimized Portfolio 2025")
plt.plot(np.arange(len(sp500_cumulative_return)),sp500_cumulative_return,label="S&P 500")

plt.xlabel("Trading Days")
plt.ylabel("Cumulative Returns")
plt.title("Optimal Portofolio 2025 vs S&P 500")
plt.legend()
plt.xlim(left=0)
plt.show()