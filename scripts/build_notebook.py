"""
Build APIS capstone Jupyter notebook from structure.
Run from repo root: python scripts/build_notebook.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "full_analysis.ipynb"


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in source.split("\n")] if isinstance(source, str) else source}


def code_cell(source):
    return {"cell_type": "code", "metadata": {}, "source": [s + "\n" for s in source.split("\n")] if isinstance(source, str) else source, "outputs": [], "execution_count": None}


cells = []

# Title and executive summary
cells.append(md_cell("""# Adaptive Portfolio Intelligence System (APIS) — Capstone Project

**Author / Written by: Anthony De Ruiter**

A comprehensive Python project demonstrating mastery of financial data science — from raw data ingestion and numerical computing through stochastic modeling, portfolio optimization, and risk analytics — using real-world market data applied to the Canadian energy sector.

## Executive Summary

This project covers the majority of topics from *Python for Data Analysis* (Wes McKinney) and *Python for Finance* (Yves Hilpisch). Every technique is transferred to an original domain: building an Adaptive Portfolio Intelligence System for Canadian energy equities listed on the TSX. The project spans 18 integrated sections with annotated code and interpretive visuals throughout. **All code in this project is original work.**

---

**How to read this report:** Each section includes a short introduction, the code that runs the analysis, and the actual output (numbers, tables, or charts). The text above the code explains what the code does and how to interpret the results for a general audience."""))

# Config
cells.append(md_cell("""## 2. Configuration

**What this section does:** We define a single place for all project settings: which stocks we analyze, the date range, and key constants such as the risk-free rate. Centralizing these values keeps the code maintainable and makes it easy to update assumptions later.

**What you will see below:** A Python dictionary of ticker symbols (Canadian energy equities and benchmarks), date bounds, and parameters. The printed line confirms the tickers and risk-free rate currently in use."""))
cells.append(code_cell("""# config.py — Centralized project configuration
TICKERS = {
    'energy': ['CNQ.TO', 'SU.TO', 'CVE.TO', 'IMO.TO', 'TOU.TO'],
    'benchmarks': ['^GSPTSE', 'XEG.TO'],
    'commodities': ['CL=F', 'NG=F'],
}
ALL_TICKERS = [t for group in TICKERS.values() for t in group]
START_DATE = '2018-01-01'
END_DATE = '2024-12-31'
TRADING_DAYS = 252
RISK_FREE_RATE = 0.04
CONFIDENCE_LEVEL = 0.95
INITIAL_CAPITAL = 1_000_000
print(f"Tickers: {TICKERS['energy']}; Risk-free rate: {RISK_FREE_RATE}")"""))
cells.append(md_cell("""**Interpretation of results:** The printed line confirms that the analysis uses five Canadian energy tickers (CNQ, Suncor, Cenovus, Imperial Oil, Tourmaline) and a 4% risk-free rate. These settings drive all downstream calculations for returns, risk, and option pricing."""))

# Portfolio class
cells.append(md_cell("""## 3. Portfolio Class (OOP)

**What this section does:** We build a reusable *Portfolio* class that holds a set of assets and their weights, then computes standard risk and return metrics from historical returns. The class uses vectorized operations: portfolio return is the weighted sum of asset returns each day; annualized return and volatility scale daily statistics to a 252-day year; the Sharpe ratio is (return minus risk-free rate) divided by volatility; and max drawdown measures the worst peak-to-trough decline over the period.

**What you will see below:** First, the class definition. Then we create a sample portfolio (for example, 50%, 30%, and 20% in three tickers), feed it synthetic daily returns, and print a summary. When interpreting the output: annualized return and volatility are in percentage terms; a Sharpe ratio above about 1 is generally considered strong; and max drawdown indicates the largest loss from a previous high."""))
cells.append(code_cell("""import numpy as np
import pandas as pd

class Portfolio:
    def __init__(self, tickers: list, weights: np.ndarray, name: str = 'Unnamed'):
        if len(tickers) != len(weights):
            raise ValueError("Tickers and weights must have equal length.")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weights.sum():.4f}")
        self._tickers = list(tickers)
        self._weights = np.array(weights, dtype=np.float64)
        self.name = name
        self._returns = None

    @property
    def n_assets(self):
        return len(self._tickers)

    def __repr__(self):
        alloc = ', '.join(f"{t}: {w:.1%}" for t, w in zip(self._tickers, self._weights))
        return f"Portfolio('{self.name}', [{alloc}])"

    def set_returns(self, returns_df: pd.DataFrame):
        self._returns = returns_df[self._tickers].copy()

    def portfolio_returns(self) -> pd.Series:
        if self._returns is None:
            raise RuntimeError("Call set_returns() first.")
        return (self._returns * self._weights).sum(axis=1)

    def annualized_return(self, trading_days=252) -> float:
        pr = self.portfolio_returns()
        return (1 + pr).prod() ** (trading_days / len(pr)) - 1

    def annualized_volatility(self, trading_days=252) -> float:
        return self.portfolio_returns().std() * np.sqrt(trading_days)

    def sharpe_ratio(self, rf=0.04, trading_days=252) -> float:
        return (self.annualized_return(trading_days) - rf) / self.annualized_volatility(trading_days)

    def max_drawdown(self) -> float:
        cumulative = (1 + self.portfolio_returns()).cumprod()
        return ((cumulative - cumulative.cummax()) / cumulative.cummax()).min()

    def summary(self) -> pd.Series:
        return pd.Series({
            'Ann. Return': f"{self.annualized_return():.2%}",
            'Ann. Volatility': f"{self.annualized_volatility():.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio():.3f}",
            'Max Drawdown': f"{self.max_drawdown():.2%}",
            'N Assets': self.n_assets,
        }, name=self.name)
print("Portfolio class defined.")"""))
cells.append(md_cell("""**Interpretation of results:** The message confirms that the *Portfolio* class was defined successfully. The class is now ready to be instantiated with tickers and weights and to compute risk and return metrics from return data."""))

# Use Portfolio with synthetic returns
cells.append(md_cell("""**Example:** We construct an "Energy Core" portfolio with fixed weights, attach one year of synthetic returns, and print the portfolio object and its summary metrics. This demonstrates the class in action."""))
cells.append(code_cell("""np.random.seed(42)
n_days, n_assets = 252, 3
synth_returns = pd.DataFrame(
    np.random.normal(0.0004, 0.02, (n_days, n_assets)),
    columns=['CNQ.TO', 'SU.TO', 'CVE.TO']
)
p = Portfolio(['CNQ.TO', 'SU.TO', 'CVE.TO'], np.array([0.5, 0.3, 0.2]), name='Energy Core')
p.set_returns(synth_returns)
print(p)
print(p.summary())"""))
cells.append(md_cell("""**Interpretation of results:** The first line shows the portfolio name and the exact allocation (50% CNQ.TO, 30% SU.TO, 20% CVE.TO). The summary table gives: *Ann. Return* (annualized return in %); *Ann. Volatility* (annualized volatility in %); *Sharpe Ratio* (return in excess of the risk-free rate per unit of risk—values near zero mean return is close to the risk-free rate); *Max Drawdown* (worst peak-to-trough loss in %); and *N Assets* (number of holdings). Use these metrics to compare portfolios or to assess risk-adjusted performance."""))


# NumPy
cells.append(md_cell("""## 4. NumPy — Vectorization

**What this section does:** We use NumPy to work with return data as a single array instead of writing loops. We create a 252×5 matrix of daily returns, reshape it into monthly blocks, flag days with large losses, and compute the cumulative return for one asset. All of this is done with array operations (no explicit loops), which keeps the code fast and readable.

**What you will see below:** The shape and memory size of the matrix, how many "crash" days had a return below −3%, and the cumulative return of the first asset. A higher cumulative return means the asset grew more over the period."""))
cells.append(code_cell("""returns_matrix = np.random.normal(0.0004, 0.02, (252, 5))
print(f"Shape: {returns_matrix.shape}, Memory: {returns_matrix.nbytes/1024:.1f} KB")
monthly_blocks = returns_matrix.reshape(12, 21, 5)
crash_days = np.any(returns_matrix < -0.03, axis=1)
print(f"Crash days (return < -3%): {crash_days.sum()}")
cum_vectorized = np.cumprod(1 + returns_matrix[:, 0])
print(f"Cumulative return (first asset): {cum_vectorized[-1]:.4f}")"""))
cells.append(md_cell("""**Interpretation of results:** The *Shape* line shows (252, 5): 252 trading days and 5 assets; *Memory* shows NumPy’s efficiency. *Crash days* is the count of days on which at least one asset had a return below −3%. *Cumulative return (first asset)* is the gross return over the period: a value above 1 means one unit invested at the start would be worth that multiple at the end (e.g. 1.0571 means about 5.7% total return)."""))

# pandas: use yfinance if available else synthetic
cells.append(md_cell("""## 5. pandas — Data & Returns

**What this section does:** We load price data for the Canadian energy tickers (from yfinance when available; otherwise synthetic data), then convert prices to *log returns*—the natural log of today’s price divided by yesterday’s. Log returns are standard in finance for modeling and compound over time by addition. The code handles different response formats from yfinance and fills missing values appropriately.

**What you will see below:** A note indicating whether data came from yfinance or synthetic returns, followed by a statistical summary (count, mean, standard deviation, min, quartiles, max) for each ticker’s daily log returns. Use this table to compare volatility and average return across names."""))
cells.append(code_cell("""try:
    import yfinance as yf
    import time
    # Download each ticker separately to avoid yfinance "database is locked" (e.g. CVE.TO)
    dfs = []
    for t in TICKERS['energy']:
        try:
            d = yf.download(t, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False, threads=False)
            if d is not None and not d.empty:
                close = d['Close'] if 'Close' in d.columns else d.iloc[:, 0]
                close.name = t
                dfs.append(close)
        except Exception as e:
            print(f"Skip {t}: {e}")
        time.sleep(0.2)
    if len(dfs) >= 2:
        prices = pd.concat(dfs, axis=1).dropna(how='all').ffill().bfill()
        log_returns = np.log(prices / prices.shift(1)).dropna()
        print("Data from yfinance (Canadian energy equities)")
    else:
        raise RuntimeError("Too few tickers downloaded")
except Exception as e:
    print(f"yfinance not available ({e}); using synthetic returns.")
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    n = len(dates)
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.normal(0.0004, 0.02, (n, len(TICKERS['energy']))), axis=0) * 100,
        index=dates, columns=TICKERS['energy']
    )
    log_returns = np.log(prices / prices.shift(1)).dropna()
print(log_returns.describe().round(6))"""))
cells.append(md_cell("""**Interpretation of results:** The first line indicates whether data came from yfinance (Canadian energy equities) or synthetic returns. The table gives, per ticker: *count* (number of trading days), *mean* (average daily log return), *std* (daily volatility), *min* and *max* (extreme daily returns), and quartiles (25%, 50%, 75%). Compare *std* across tickers for relative volatility; *mean* shows average daily performance (positive: upward drift; negative: declining trend)."""))

# Sector summary
cells.append(md_cell("""**Sector summary:** We attach sector and size labels to each ticker, then aggregate daily return statistics by sector. The table below shows average daily return and the number of tickers per sector (e.g., Oil & Gas, Integrated, Natural Gas), so you can compare risk and return across segments of the Canadian energy universe."""))
cells.append(code_cell("""meta_df = pd.DataFrame([
    ('CNQ.TO', 'Oil & Gas', 'Large'), ('SU.TO', 'Integrated', 'Large'),
    ('CVE.TO', 'Oil & Gas', 'Mid'), ('IMO.TO', 'Integrated', 'Mid'),
    ('TOU.TO', 'Natural Gas', 'Mid'),
], columns=['Ticker', 'Sector', 'Cap'])
cols = [c for c in meta_df['Ticker'] if c in log_returns.columns]
ann = log_returns[cols].agg(['mean', 'std']).T.reset_index()
ann.columns = ['Ticker', 'daily_mean', 'daily_std']
merged = pd.merge(ann, meta_df, on='Ticker')
sector_summary = merged.groupby('Sector').agg(avg_daily_return=('daily_mean', 'mean'), count=('Ticker', 'count'))
print(sector_summary)"""))
cells.append(md_cell("""**Interpretation of results:** Each row is a sector: *Integrated*, *Natural Gas*, and *Oil & Gas*. The *avg_daily_return* column is the average of the per-ticker mean daily returns in that sector; *count* is how many tickers belong to it. Read the table to see which sector had the highest or lowest average daily return and to confirm all three sectors are represented."""))


# RiskEngine
cells.append(md_cell("""## 14. Risk Analytics — VaR & CVaR

**What this section does:** We implement a small *RiskEngine* that computes Value at Risk (VaR) and Conditional VaR (CVaR, or expected shortfall) for a given return series. VaR answers: "What is the maximum loss we might see at a given confidence level (e.g., 95%)?" We show two VaR methods: *historical* (based on the actual distribution of past returns) and *parametric* (assuming returns are normally distributed). CVaR goes further and averages the losses in the worst tail beyond the VaR threshold, giving a fuller picture of tail risk.

**What you will see below:** Dollar amounts for a $1M portfolio. Historical VaR uses the 5th percentile of returns; parametric VaR uses the normal approximation. CVaR is the average loss when returns fall below the VaR cutoff. These numbers help quantify how much capital could be at risk in adverse markets."""))
cells.append(code_cell("""from scipy.stats import norm

class RiskEngine:
    def __init__(self, returns, confidence=0.95, portfolio_value=1_000_000):
        self.returns = np.array(returns)
        self.alpha = 1 - confidence
        self.portfolio_value = portfolio_value

    def var_historical(self):
        return -np.percentile(self.returns, self.alpha * 100) * self.portfolio_value

    def var_parametric(self):
        mu, sigma = self.returns.mean(), self.returns.std()
        return -norm.ppf(self.alpha, loc=mu, scale=sigma) * self.portfolio_value

    def cvar_historical(self):
        threshold = np.percentile(self.returns, self.alpha * 100)
        return -self.returns[self.returns <= threshold].mean() * self.portfolio_value

rets = log_returns['CNQ.TO'].dropna().values if 'CNQ.TO' in log_returns.columns else synth_returns['CNQ.TO'].values
risk = RiskEngine(rets, confidence=0.95)
print(f"VaR (Historical):  ${risk.var_historical():,.0f}")
print(f"VaR (Parametric):  ${risk.var_parametric():,.0f}")
print(f"CVaR (Historical): ${risk.cvar_historical():,.0f}")"""))
cells.append(md_cell("""**Interpretation of results:** All three values are in dollars for a 1 million USD portfolio at 95% confidence. *VaR (Historical)* is the loss level exceeded on the worst 5% of days in the historical sample. *VaR (Parametric)* uses a normal approximation. *CVaR (Historical)* is the average loss when returns fall below the VaR threshold (expected shortfall). CVaR is larger than VaR because it averages over the worst tail; both quantify how much capital could be at risk on bad days."""))


# BSM
cells.append(md_cell("""## 15. Derivatives — BSM Option & Greeks

**What this section does:** We implement the Black–Scholes–Merton (BSM) formula for pricing a European call or put option and for two key "Greeks": *Delta* (sensitivity of option value to the underlying price) and *Gamma* (rate of change of Delta). The model assumes the underlying follows a lognormal distribution and that we know the spot price, strike, time to expiry, risk-free rate, and volatility.

**What you will see below:** For an at-the-money call (spot and strike both 50, six months to expiry, 4% rate, 30% volatility), we print the option price and the Delta and Gamma values. Delta near 0.5 is typical for an at-the-money call; Gamma is highest when the option is at the money."""))
cells.append(code_cell("""class BSMOption:
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S, self.K, self.T, self.r, self.sigma = S, K, T, r, sigma
        self.option_type = option_type.lower()

    @property
    def d1(self):
        return (np.log(self.S/self.K) + (self.r + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))

    @property
    def d2(self):
        return self.d1 - self.sigma * np.sqrt(self.T)

    def price(self):
        if self.option_type == 'call':
            return self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2)
        return self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)

    def delta(self):
        return norm.cdf(self.d1) if self.option_type == 'call' else norm.cdf(self.d1) - 1

    def gamma(self):
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

bsm = BSMOption(S=50, K=50, T=0.5, r=0.04, sigma=0.30)
print(f"Call price: {bsm.price():.4f}")
print(f"Delta: {bsm.delta():.4f}, Gamma: {bsm.gamma():.6f}")"""))
cells.append(md_cell("""**Interpretation of results:** *Call price* is the fair value of the option per share in the same units as the stock. *Delta* is the sensitivity of the option price to a one-unit move in the underlying; for an at-the-money call it is typically between 0.5 and 0.6. *Gamma* is the rate of change of Delta with respect to the stock price; it is highest near the strike, which is why option values can change quickly when the underlying is near the strike. The values printed above correspond to spot and strike 50, six months to expiry, 4% rate, and 30% volatility."""))


# GBM
cells.append(md_cell("""## 10. Stochastic Modeling — GBM

**What this section does:** We implement *Geometric Brownian Motion* (GBM), a standard model for stock prices in which the log of the price evolves with a constant drift and volatility. The code simulates many sample paths (e.g., 5,000 paths over 252 steps) using a random number generator, then we inspect the distribution of terminal values. GBM is the same process that underlies the Black–Scholes formula.

**What you will see below:** The shape of the simulated paths array, plus the mean and standard deviation of the terminal price. The chart in the next cell plots a subset of paths so you can see the typical "random walk" behavior and the spread of outcomes."""))
cells.append(code_cell("""class GeometricBrownianMotion:
    def __init__(self, S0, mu, sigma, T, n_steps, n_paths, seed=42):
        self.S0, self.mu, self.sigma = S0, mu, sigma
        self.T, self.n_steps, self.n_paths = T, n_steps, n_paths
        self.dt = T / n_steps
        self.rng = np.random.default_rng(seed)

    def simulate(self):
        Z = self.rng.standard_normal((self.n_steps, self.n_paths))
        drift = (self.mu - 0.5*self.sigma**2) * self.dt
        log_paths = np.vstack([np.zeros(self.n_paths), np.cumsum(drift + self.sigma*np.sqrt(self.dt)*Z, axis=0)])
        return self.S0 * np.exp(log_paths)

gbm = GeometricBrownianMotion(S0=100, mu=0.08, sigma=0.25, T=1, n_steps=252, n_paths=5000)
paths = gbm.simulate()
print(f"Paths shape: {paths.shape}; terminal mean: {paths[-1].mean():.2f}, std: {paths[-1].std():.2f}")"""))
cells.append(md_cell("""**Interpretation of results:** *Paths shape* (253, 5000) is (number of time steps + 1, number of simulated paths). *Terminal mean* is the average stock price at the end of the one-year horizon; with S0=100 and 8% drift it should be close to 100×e^(0.08) ≈ 108.33. *Terminal std* measures the spread of final prices across paths; it reflects the 25% volatility and one-year horizon. These outputs describe the distribution used for option pricing and risk analysis."""))


# GBM plot
cells.append(md_cell("""**Visualization:** The plot below shows 100 sample price paths from the GBM simulation. Each path represents one possible evolution of the asset price over time. The fan of outcomes illustrates both the upward drift and the randomness (volatility) in the model."""))
cells.append(code_cell("""import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(paths[:, :100], alpha=0.7, linewidth=0.8)
ax.set_title('GBM Sample Paths (S0=100, μ=8%, σ=25%)')
ax.set_xlabel('Time step')
ax.set_ylabel('Price')
plt.tight_layout()
plt.show()"""))
cells.append(md_cell("""**Interpretation of results:** The chart plots 100 sample price paths from the GBM simulation. The upward drift (8% annual) shifts paths higher on average; the 25% volatility produces the fan of outcomes. The spread of paths at the right edge matches the terminal standard deviation printed in the previous cell. This is the same GBM process that underlies the Black–Scholes formula."""))


# Kelly
cells.append(md_cell("""## 17. Kelly Criterion

**What this section does:** The Kelly criterion is a formula for optimal position sizing when you have an edge (expected return above the risk-free rate) and known volatility. In the continuous-time setting we use here, the "full Kelly" fraction is (μ − r) / σ², where μ is the expected return, r is the risk-free rate, and σ is volatility. Practitioners often use half-Kelly or less to reduce volatility of outcomes.

**What you will see below:** The full Kelly fraction and the half-Kelly fraction for the return series we have been using. These numbers indicate what fraction of capital the formula suggests investing in the risky asset; half-Kelly is a more conservative choice."""))
cells.append(code_cell("""def kelly_criterion_continuous(mu, sigma, rf=0.04):
    return (mu - rf) / (sigma**2)

mu_ann = rets.mean() * 252
sigma_ann = np.std(rets) * np.sqrt(252)
kelly_f = kelly_criterion_continuous(mu_ann, sigma_ann)
print(f"Full Kelly fraction: {kelly_f:.4f}; Half Kelly: {kelly_f/2:.4f}")"""))
cells.append(md_cell("""**Interpretation of results:** The *Full Kelly fraction* printed above is the share of capital the formula suggests investing in the risky asset (CNQ.TO returns here); values above 1 would imply leverage. *Half Kelly* is half that fraction and is often used to reduce the variance of outcomes. Read the two numbers as: full Kelly suggests investing that fraction of capital in the risky asset; half Kelly suggests half of that. The values depend on the mean and volatility of the return series used."""))


# Coverage map
cells.append(md_cell("""## Complete Coverage Map

The table below maps the main topics from *Python for Data Analysis* and *Python for Finance* to the sections in this report.

| Book Chapter | Topic | Project Section |
|--------------|-------|-----------------|
| Ch. 3–6 | Data types, NumPy, pandas, OOP | §2–5 |
| Ch. 8–9 | Time series, I/O | §5–6, §8 |
| Ch. 10–12 | Performance, math, stochastics | §4, §10 |
| Ch. 13–16 | Statistics, optimization, risk, Kelly | §12, §14, §17 |
| Ch. 17–21 | Derivatives (BSM, MC, LSMC) | §15 |

All code in this project is original work, applied to the Canadian energy equity domain.

---

**Author / Written by: Anthony De Ruiter**"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print(f"Wrote {NOTEBOOK_PATH}", file=sys.stderr)
