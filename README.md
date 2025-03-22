# 📈 Factor Model & Smart Beta Portfolio Builder

[![Railway App](https://img.shields.io/badge/Railway-Deployed-success)](https://smart-beta-portfolio-builder.up.railway.app/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![GitHub stars](https://img.shields.io/github/stars/jeevanba273/Factor-Model-and-Smart-Beta-Portfolio-Builder?style=social)](https://github.com/jeevanba273/Factor-Model-and-Smart-Beta-Portfolio-Builder)
![GitHub forks](https://img.shields.io/github/forks/jeevanba273/Factor-Model-and-Smart-Beta-Portfolio-Builder?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/jeevanba273/Factor-Model-and-Smart-Beta-Portfolio-Builder)
![GitHub repo size](https://img.shields.io/github/repo-size/jeevanba273/Factor-Model-and-Smart-Beta-Portfolio-Builder?color=blue&style=flat-square)


> **Live App**: [Factor Model & Smart Beta Portfolio Builder on Railway](https://smart-beta-portfolio-builder.up.railway.app/)

---

## 📑 Table of Contents
1. [🚀 Introduction](#-introduction)  
2. [✨ Features](#-features)  
   1. [📊 Factor Analysis](#-factor-analysis)  
   2. [🏗 Portfolio Construction](#-portfolio-construction)  
   3. [📉 Comprehensive Backtesting](#-comprehensive-backtesting)  
   4. [📂 Portfolio Analysis](#-portfolio-analysis)  
3. [🛠 How It Works](#-how-it-works)  
   1. [Factor Calculation Methodology](#factor-calculation-methodology)  
   2. [Portfolio Construction Process](#portfolio-construction-process)  
   3. [Backtest Implementation](#backtest-implementation)  
4. [📋 Using the Application](#-using-the-application)  
5. [📊 Performance Metrics Explained](#-performance-metrics-explained)  
6. [📈 Factor Investment Benefits](#-factor-investment-benefits)  
7. [📋 Data Sources and Processing](#-data-sources-and-processing)  
8. [💻 Technical Implementation](#-technical-implementation)  
9. [🔍 Investment Insights](#-investment-insights)  
10. [🚀 Installation & Local Development](#-installation--local-development)  
11. [🔗 Railway App Deployment](#-railway-app-deployment)  
12. [👤 Author](#-author)  
13. [🙏 Acknowledgements](#-acknowledgements)  

---

## 🚀 Introduction
The **Indian Market Factor Model & Smart Beta Portfolio Builder** is a sophisticated financial analysis tool designed for investors interested in applying factor-based investing strategies to the Indian stock market. This application enables users to analyze factor exposures, build custom smart beta portfolios, and backtest investment strategies against historical data.

Unlike traditional market-cap weighted indices, factor-based investing (also known as *smart beta*) allows investors to target specific drivers of returns such as value, momentum, low volatility, and quality. This application brings institutional-grade portfolio construction tools to individual investors through an intuitive, interactive interface.

---

## ✨ Features

### 📊 Factor Analysis
- **Multi-factor Model**: Analyze stocks through the lens of four proven investment factors:
  - **Value**: Identifies undervalued stocks based on price-to-volume metrics  
  - **Momentum**: Captures stocks with positive price trends (12-month minus 1-month returns)  
  - **Low Volatility**: Favors stocks with lower price fluctuations for more stable returns  
  - **Quality**: Focuses on stocks with stable trading patterns and stronger investor holdings  
- **Factor Correlation Analysis**: Visualize relationships between factors to understand diversification benefits  
- **Top Stock Rankings**: Identify the highest-ranking stocks for each factor  

### 🏗 Portfolio Construction
- **Custom Factor Weights**: Adjust the importance of each factor with intuitive sliders  
- **Equal-Weight Reset**: Quick option to reset all factors to equal weights  
- **Portfolio Size Control**: Select the number of stocks to include (e.g., 10–100)  
- **Rebalancing Frequency**: Choose between monthly, quarterly, or annual rebalancing  

### 📉 Comprehensive Backtesting
- **Custom Date Ranges**: Test factor strategies across different market cycles  
- **Performance Metrics**: Analyze your portfolio with institutional-grade metrics:
  - CAGR (Compound Annual Growth Rate)  
  - Volatility (Annualized Standard Deviation)  
  - Sharpe Ratio (Risk-Adjusted Returns)  
  - Sortino Ratio (Downside Risk-Adjusted Returns)  
  - Maximum Drawdown  
  - Drawdown Duration  
- **Benchmark Comparison**: Compare your strategy against the NIFTY 50 index  
- **Interactive Charts**: Visualize performance and drawdowns with dynamic Plotly charts  

### 📂 Portfolio Analysis
- **Current Holdings**: View a detailed breakdown of portfolio constituents  
- **Factor Exposure**: Analyze your portfolio’s exposure to each factor  
- **Export Functionality**: Download your portfolio for further analysis or implementation  

---

## 🛠 How It Works

### Factor Calculation Methodology

#### Value Factor
Identifies potentially undervalued stocks using a price-to-volume ratio as a proxy for traditional value metrics.  
- **Price-to-Volume Ratio** = Close Price ÷ (Net Traded Value ÷ Net Traded Quantity)  
- **Value Score** = 1 – Rank(Price-to-Volume Ratio)  

A lower price-to-volume ratio translates into a higher Value Score.

#### Momentum Factor
Captures stocks with strong positive price trends using a classic 12-month return minus the most recent 1-month return to avoid short-term reversals.  
- **Momentum** = 12-Month Return – 1-Month Return  
- **Momentum Score** = Rank(Momentum)  

A higher momentum value corresponds to a stronger price trend.

#### Low Volatility Factor
Prioritizes stocks with more stable (lower) price fluctuations.  
- **Volatility Score** = 1 – Rank(Standard Deviation of Daily Returns)  

Lower standard deviation leads to a higher factor score.

#### Quality Factor
Combines trading stability and delivery volume ratio to identify stocks with potentially stronger fundamentals.  
- **Trading Volatility** = (Std. Deviation of Traded Quantity) ÷ (Mean of Traded Quantity)  
- **Delivery Ratio** = (Total Delivery Volume) ÷ (Total Traded Quantity)  
- **Quality Score** = 0.5 × (1 – Rank(Trading Volatility)) + 0.5 × Rank(Delivery Ratio)  

Stocks with stable trading patterns and higher delivery volume receive higher Quality Scores.

---

### Portfolio Construction Process
1. **Factor Score Calculation**: Each stock receives a 0–1 score for each active factor.  
2. **Composite Score Generation**: Factor scores are weighted according to user preferences:

   Composite Score =  
   (Value Score × Value Weight) +  
   (Momentum Score × Momentum Weight) +  
   (Volatility Score × Volatility Weight) +  
   (Quality Score × Quality Weight)

3. **Stock Selection**: The top *N* stocks (based on the composite score) are selected.  
4. **Equal Weighting**: Each selected stock is assigned an equal weight in the portfolio (1/N).  
5. **Rebalancing**: The portfolio is rebalanced at the selected frequency (monthly, quarterly, annually).  

---

### Backtest Implementation
1. **Data Subsetting**: Historical data is filtered to the chosen date range.  
2. **Rebalancing Dates**: Determined based on the selected rebalancing frequency.  
3. **Portfolio Simulation**:  
   - Recalculate factor scores at each rebalance date (using data available at that time).  
   - Rebalance the portfolio using updated factor scores.  
   - Track performance until the next rebalance date.  
4. **Performance Calculation**: Portfolio returns, drawdowns, and risk metrics are computed.  
5. **Benchmark Comparison**: Portfolio performance is compared to the NIFTY 50 index.  

---

## 📋 Using the Application

1. **Select Factors**  
   In the sidebar, enable or disable specific factors (Value, Momentum, Low Volatility, Quality).

2. **Adjust Factor Weights**  
   Use sliders to set the relative importance of each active factor. Weights automatically normalize to 1. Click “Reset to Equal Weights” for a quick balance.

3. **Configure Portfolio Settings**  
   - Choose how many stocks to include (e.g., 10–100).  
   - Select rebalancing frequency (monthly, quarterly, annual).  
   - Specify the backtest period via date pickers.

4. **Run Analysis**  
   Click “Run Analysis” to execute the factor model and backtest.

5. **Explore Results**  
   - **Factor Analysis Tab**: Check factor correlations and top stocks per factor.  
   - **Portfolio Backtest Tab**: View performance charts, metrics, and benchmark comparisons.  
   - **Current Portfolio Tab**: See detailed holdings and factor exposures.

6. **Export Portfolio**  
   Download your portfolio as a CSV for further analysis or live market execution.

---

## 📊 Performance Metrics Explained

### CAGR (Compound Annual Growth Rate)
Annualized rate of return representing the geometric progression of portfolio value over time:

    CAGR = (Final Value / Initial Value)^(1 / Number of Years) - 1

### Volatility
Annualized standard deviation of returns, indicating how widely returns fluctuate:

    Volatility = StdDev(Daily Returns) × √252

*(Assumes ~252 trading days in a year.)*

### Sharpe Ratio
Risk-adjusted return per unit of total volatility. A higher Sharpe ratio indicates better risk-adjusted performance:

    Sharpe Ratio = [(Average Daily Return) - (Risk-Free Rate)] / StdDev(Daily Returns) × √252

### Sortino Ratio
Similar to Sharpe but focuses only on downside volatility, providing a better measure for investors concerned primarily about losses:

    Sortino Ratio = [(Average Daily Return) - (Risk-Free Rate)] / Downside Deviation × √252

### Maximum Drawdown
Measures the largest peak-to-trough decline during the backtest period:

    Max Drawdown = (Trough Value - Peak Value) / Peak Value

### Drawdown Duration
Time between reaching a peak and returning to that same peak level (often measured in days).

---

## 📈 Factor Investment Benefits

1. **Systematic Approach**: Reduces behavioral biases through a rules-based methodology.  
2. **Targeted Exposure**: Allows tilting the portfolio toward specific return drivers.  
3. **Risk Management**: Helps manage and diversify risk by understanding factor exposures.  
4. **Enhanced Returns**: Academic studies suggest certain factor tilts may outperform cap-weighted benchmarks long-term.

---

## 📋 Data Sources and Processing

### Data Sources
- Historical price data (Open, High, Low, Close)  
- Volume data (Traded Quantity)  
- Delivery data (Delivery Percentage)  

All data comes from the National Stock Exchange (NSE) of India and is adjusted for corporate actions (splits, bonus and rights).

### Data Processing
- **Date Range Selection**: Users can limit analyses to specific time frames.  
- **Performance Optimization**: Optionally reduce dataset size for quicker computation.  
- **Lookback Periods**: Automatically calculated for factor computation (e.g., 12-month lookback for Momentum).  
- **Data Validation**: Ensures data availability for all selected stocks and periods.

---

## 💻 Technical Implementation

- **Streamlit**: Interactive web framework for data apps.  
- **Pandas / NumPy**: Core libraries for data manipulation and numerical computing.  
- **Plotly**: Interactive charting library.  
- **StatsModels**: Advanced statistical analysis (optional usage).  

Key elements:  
- **Memory-Efficient Processing**: Uses chunked data operations where possible.  
- **Caching**: Utilizes Streamlit’s cache for heavy computations.  
- **Vectorized Operations**: Pandas-based speed improvements.  
- **Interactive UI**: Dynamic updates without full page reload.  
- **Railway Deployment**: Seamless hosting, with automatic scaling.

---

## 🔍 Investment Insights

1. **Factor Rotation**: Different factors tend to lead in different market phases (Value in recovery, Quality in downturns, etc.).  
2. **Factor Blending**: Combining multiple factors can smooth returns and reduce cyclicality.  
3. **Rebalancing Frequency**: Higher frequency captures factor changes sooner but increases transaction costs.  
4. **Indian Market Nuances**: Higher retail participation, sector concentration, and unique regulations can impact factor performance.

---

## 🚀 Installation & Local Development

To run this application on your local machine:

1. **Clone the Repository**  

    ```bash
    git clone https://github.com/jeevanba273/Factor-Model-and-Smart-Beta-Portfolio-Builder.git  
    cd Factor-Model-and-Smart-Beta-Portfolio-Builder  

2. **Install Required Packages**  

    ```bash
    pip install -r requirements.txt  

3. **Run the Streamlit App**  

    ```bash
    streamlit run app.py  

After this, your browser should open automatically with the application running locally.

---

## 🔗 Railway App Deployment

This application is deployed on [Railway](https://railway.app/), offering:

- **24/7 Availability**: Access the app anytime, anywhere.  
- **Scalable Resources**: Capable of handling significant computational loads.  
- **Secure Environment**: Safeguards your financial data and analysis.  
- **Optimized Performance**: Fast loading times for factor calculations.

Visit the live app here: [Indian Market Factor Model on Railway](https://smart-beta-portfolio-builder.up.railway.app/)

---

## 👤 Author

Created by **Jeevan B A**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Jeevan_B_A-blue)](https://www.linkedin.com/in/jeevanba273/)

---

## 🙏 Acknowledgements

- **National Stock Exchange of India** for market data  
- **Streamlit** for the excellent data app framework  
- **Railway** for the hosting platform  

---

> ⭐ If you find this project useful, please [star the repository on GitHub](https://github.com/jeevanba273/Factor-Model-and-Smart-Beta-Portfolio-Builder)! ⭐
