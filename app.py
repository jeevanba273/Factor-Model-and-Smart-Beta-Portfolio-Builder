import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import gc
import time
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Inject custom CSS to center ONLY the date picker without affecting dropdowns
st.markdown(
    """
    <style>
    /* More specific selector for date picker popups */
    div[data-baseweb="datepicker"] div[data-baseweb="popover"] {
        left: 50% !important;
        transform: translateX(-50%) !important;
    }
    
    /* Target calendar container more specifically */
    div[data-baseweb="datepicker"] div[role="dialog"] {
        left: 50% !important;
        transform: translateX(-50%) !important;
    }
    
    /* Custom tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #ccc;
        cursor: help;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 250px;
        background-color: #f9f9f9;
        color: #333;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -125px;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0px 0px 8px rgba(0,0,0,0.1);
        font-size: 0.8em;
        line-height: 1.4;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper function to create tooltips
def tooltip(text, tooltip_text):
    return f"""
    <div class="tooltip">{text}
        <span class="tooltiptext">{tooltip_text}</span>
    </div>
    """

# App title and description
st.title("Indian Market Factor Model & Smart Beta Portfolio Builder")
st.write("Analyze factor exposures and build custom factor-based portfolios using Indian stock market data")

# Memory-efficient data loading with chunking
@st.cache_data
def load_data():
    try:
        # Try different possible paths for different platforms
        possible_paths = [
            "data/final_adjusted_stock_data.csv",
            "./data/final_adjusted_stock_data.csv", 
            "../data/final_adjusted_stock_data.csv",
            "/app/data/final_adjusted_stock_data.csv"
        ]
        
        # Debug information to see what's happening
        current_dir = os.getcwd()
        st.write(f"Current working directory: {current_dir}")
        available_dirs = os.listdir()
        st.write(f"Available directories: {available_dirs}")
        
        # Check if any of the possible paths exist
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                st.write(f"Found data file at: {file_path}")
                break
        
        if file_path is None:
            st.error("Data file not found. Please check if the file exists and is in the correct location.")
            return pd.DataFrame()
            
        # Continue with your existing chunking logic
        file_size_gb = os.path.getsize(file_path) / (1024 * 1024 * 1024)
        
        if file_size_gb > 0.5:  # If file is larger than 500MB
            # Use chunking for large files
            chunk_size = 500000  # Adjust based on available memory
            chunks = []
            
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                chunk['DATE'] = pd.to_datetime(chunk['DATE'])
                chunks.append(chunk)
                
                # Force garbage collection after processing each chunk
                gc.collect()
                
            df = pd.concat(chunks, ignore_index=True)
            # Force garbage collection after concatenation
            gc.collect()
            return df
        else:
            # For smaller files, load directly
            df = pd.read_csv(file_path, low_memory=False)
            df['DATE'] = pd.to_datetime(df['DATE'])
            return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Show the full error to debug
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()

# Calculate returns
@st.cache_data
def calculate_returns(df):
    # Group by symbol and sort by date
    returns_df = df.sort_values(['SYMBOL', 'DATE'])
    
    # Calculate daily returns
    returns_df['DAILY_RETURN'] = returns_df.groupby('SYMBOL')['CLOSE_PRICE'].pct_change()
    
    # Calculate rolling returns (1M, 3M, 6M, 12M)
    returns_df['1M_RETURN'] = returns_df.groupby('SYMBOL')['CLOSE_PRICE'].pct_change(21)
    returns_df['3M_RETURN'] = returns_df.groupby('SYMBOL')['CLOSE_PRICE'].pct_change(63)
    returns_df['6M_RETURN'] = returns_df.groupby('SYMBOL')['CLOSE_PRICE'].pct_change(126)
    returns_df['12M_RETURN'] = returns_df.groupby('SYMBOL')['CLOSE_PRICE'].pct_change(252)
    
    return returns_df

# Factor calculation functions
@st.cache_data
def calculate_value_factor(df):
    """Calculate value factor metrics"""
    latest_data = df[df['DATE'] == df['DATE'].max()].copy()  # use copy() to avoid SettingWithCopyWarning
    
    # For this example, we'll use a simple price-to-volume ratio as a proxy
    latest_data['PRICE_TO_VOLUME'] = latest_data['CLOSE_PRICE'] / (latest_data['NET_TRDVAL'] / latest_data['NET_TRDQTY'])
    
    # Lower values are better for value factor
    latest_data['VALUE_SCORE'] = 1 - (latest_data['PRICE_TO_VOLUME'].rank(pct=True))
    
    return latest_data[['SYMBOL', 'VALUE_SCORE']]

@st.cache_data
def calculate_momentum_factor(df):
    """Calculate momentum factor based on 12-month returns minus 1-month returns"""
    latest_date = df['DATE'].max()
    momentum_df = df[df['DATE'] == latest_date][['SYMBOL', '12M_RETURN', '1M_RETURN']].copy()  # use copy() to avoid warnings
    
    # 12-month momentum skipping the most recent month
    momentum_df['MOMENTUM'] = momentum_df['12M_RETURN'] - momentum_df['1M_RETURN']
    momentum_df['MOMENTUM_SCORE'] = momentum_df['MOMENTUM'].rank(pct=True)
    
    return momentum_df[['SYMBOL', 'MOMENTUM_SCORE']]

@st.cache_data
def calculate_volatility_factor(df):
    """Calculate low volatility factor based on standard deviation of returns"""
    # Get last 252 trading days (approximately 1 year)
    last_year = df['DATE'].max() - pd.Timedelta(days=365)
    recent_data = df[df['DATE'] >= last_year]
    
    # Calculate volatility for each stock
    vol_df = recent_data.groupby('SYMBOL')['DAILY_RETURN'].std().reset_index()
    vol_df.columns = ['SYMBOL', 'VOLATILITY']
    
    # Lower volatility is better for this factor
    vol_df['VOLATILITY_SCORE'] = 1 - vol_df['VOLATILITY'].rank(pct=True)
    
    return vol_df[['SYMBOL', 'VOLATILITY_SCORE']]

@st.cache_data
def calculate_quality_factor(df):
    """
    Calculate quality factor based on:
    - Trading stability (lower volatility in traded quantity)
    - Delivery volume ratio (higher percentage delivered suggests stronger hands)
    """
    # Get last 252 trading days
    last_year = df['DATE'].max() - pd.Timedelta(days=365)
    recent_data = df[df['DATE'] >= last_year]
    
    # Calculate trading stability 
    vol_traded = recent_data.groupby('SYMBOL')['NET_TRDQTY'].std() / recent_data.groupby('SYMBOL')['NET_TRDQTY'].mean()
    vol_traded = vol_traded.reset_index()
    vol_traded.columns = ['SYMBOL', 'TRADE_STABILITY']
    
    # Calculate average delivery percentage 
    delivery_ratio = recent_data.groupby('SYMBOL')[['DELIVERY_VOLUME', 'NET_TRDQTY']].sum()
    delivery_ratio['DELIVERY_RATIO'] = delivery_ratio['DELIVERY_VOLUME'] / delivery_ratio['NET_TRDQTY']
    delivery_ratio = delivery_ratio.reset_index()[['SYMBOL', 'DELIVERY_RATIO']]
    
    # Merge the metrics
    quality_df = pd.merge(vol_traded, delivery_ratio, on='SYMBOL')
    
    # Create quality score
    # Lower trading volatility is better, higher delivery ratio is better
    quality_df['QUALITY_SCORE'] = (
        (1 - quality_df['TRADE_STABILITY'].rank(pct=True)) * 0.5 + 
        quality_df['DELIVERY_RATIO'].rank(pct=True) * 0.5
    )
    
    return quality_df[['SYMBOL', 'QUALITY_SCORE']]

@st.cache_data
def create_factor_model(df, active_factors):
    """Combine all factors into a single dataframe"""
    # Calculate returns first
    returns_df = calculate_returns(df)
    
    # Start with a dataframe containing just the symbols
    latest_data = df[df['DATE'] == df['DATE'].max()][['SYMBOL', 'CLOSE_PRICE']]
    factor_df = latest_data[['SYMBOL']].copy()
    
    # Calculate and add each active factor
    if 'value' in active_factors:
        value_df = calculate_value_factor(returns_df)
        factor_df = pd.merge(factor_df, value_df, on='SYMBOL', how='left')
    
    if 'momentum' in active_factors:
        momentum_df = calculate_momentum_factor(returns_df)
        factor_df = pd.merge(factor_df, momentum_df, on='SYMBOL', how='left')
    
    if 'volatility' in active_factors:
        volatility_df = calculate_volatility_factor(returns_df)
        factor_df = pd.merge(factor_df, volatility_df, on='SYMBOL', how='left')
    
    if 'quality' in active_factors:
        quality_df = calculate_quality_factor(returns_df)
        factor_df = pd.merge(factor_df, quality_df, on='SYMBOL', how='left')
    
    # Merge with latest prices
    factor_df = pd.merge(factor_df, latest_data, on='SYMBOL')
    
    return factor_df

def backtest_factor_portfolio(df, factor_weights, active_factors, top_n=30, start_date=None, end_date=None, rebalance_freq='monthly'):
    """
    Backtest a factor-weighted portfolio:
    - factor_weights: dict with weights for each factor
    - active_factors: list of active factors to use
    - top_n: number of stocks to include
    - start_date: beginning of backtest period
    - end_date: end of backtest period
    - rebalance_freq: 'monthly', 'quarterly', or 'annual'
    """
    if not active_factors:
        st.error("Please select at least one factor to build the portfolio.")
        return None, None, None, None
        
    # Set default dates if not provided
    if end_date is None:
        end_date = df['DATE'].max()
    if start_date is None:
        start_date = end_date - pd.Timedelta(days=365)  # Default to 1 year
    
    # Filter data to the backtest period
    backtest_data = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
    
    # Check if there's enough data in the selected period
    if len(backtest_data) < 10:  # arbitrary minimum number of data points
        st.error("Insufficient data in the selected date range. Please select a wider range.")
        return None, None, None, None
    
    # Get unique dates for rebalancing
    all_dates = sorted(backtest_data['DATE'].unique())
    
    # Get rebalancing dates based on frequency
    rebalance_dates = []
    current_month = None
    current_quarter = None
    current_year = None
    
    for date in all_dates:
        if rebalance_freq == 'monthly' and date.month != current_month:
            rebalance_dates.append(date)
            current_month = date.month
        elif rebalance_freq == 'quarterly' and (date.month-1)//3 != current_quarter:
            rebalance_dates.append(date)
            current_quarter = (date.month-1)//3
        elif rebalance_freq == 'annual' and date.year != current_year:
            rebalance_dates.append(date)
            current_year = date.year
    
    # Add the end date if it's not already included
    if all_dates[-1] != rebalance_dates[-1]:
        rebalance_dates.append(all_dates[-1])
    
    # Track absolute portfolio values (for calculating returns)
    absolute_portfolio_values = []
    holdings_history = []
    
    # For drawdown calculation
    daily_portfolio_values = []
    
    # Initial portfolio value
    portfolio_value = 100000  # Starting with 100,000 rupees
    cash = portfolio_value
    holdings = {}
    
    # Track Nifty 50 performance
    nifty_data = df[df['SYMBOL'] == 'NIFTY 50']
    
    # Store initial portfolio and Nifty values in absolute terms
    absolute_portfolio_values.append((rebalance_dates[0], portfolio_value))
    
    # Get exact Nifty value on the first rebalance date
    initial_nifty_data = nifty_data[nifty_data['DATE'] == rebalance_dates[0]]
    if len(initial_nifty_data) > 0:
        initial_nifty = initial_nifty_data.iloc[0]['CLOSE_PRICE']
    else:
        # If no exact match, get the closest previous date
        initial_nifty = nifty_data[nifty_data['DATE'] <= rebalance_dates[0]].iloc[-1]['CLOSE_PRICE']

    # For tracking the normalized values
    portfolio_values = []
    nifty_values = []
    
    # Start both at 1 (100%)
    portfolio_values.append((rebalance_dates[0], 1.0))
    nifty_values.append((rebalance_dates[0], 1.0))
    
    # Add progress bar
    progress_bar = st.progress(0)
    
    for i in range(len(rebalance_dates)-1):
        # Update progress bar on every iteration
        progress_bar.progress((i + 1) / (len(rebalance_dates)-1))
        
        current_date = rebalance_dates[i]
        next_date = rebalance_dates[i+1]
        
        # Subset data up to current date for factor calculation
        data_subset = df[df['DATE'] <= current_date]
        
        # Calculate returns
        returns_subset = calculate_returns(data_subset)
        
        # Create base dataframe with all symbols
        latest_prices = df[df['DATE'] == current_date][['SYMBOL', 'CLOSE_PRICE']]
        factor_df = latest_prices[['SYMBOL']].copy()
        
        # Calculate and add each active factor
        if 'value' in active_factors:
            value_df = calculate_value_factor(returns_subset)
            factor_df = pd.merge(factor_df, value_df, on='SYMBOL', how='left')
        
        if 'momentum' in active_factors:
            momentum_df = calculate_momentum_factor(returns_subset)
            factor_df = pd.merge(factor_df, momentum_df, on='SYMBOL', how='left')
        
        if 'volatility' in active_factors:
            volatility_df = calculate_volatility_factor(returns_subset)
            factor_df = pd.merge(factor_df, volatility_df, on='SYMBOL', how='left')
        
        if 'quality' in active_factors:
            quality_df = calculate_quality_factor(returns_subset)
            factor_df = pd.merge(factor_df, quality_df, on='SYMBOL', how='left')
        
        # Filter out NIFTY 50 from factor selection
        factor_df = factor_df[factor_df['SYMBOL'] != 'NIFTY 50']
        
        # Calculate composite score based on weights and active factors
        factor_df['COMPOSITE_SCORE'] = 0
        
        for factor in active_factors:
            factor_column = f"{factor.upper()}_SCORE"
            if factor_column in factor_df.columns:
                factor_df['COMPOSITE_SCORE'] += factor_df[factor_column] * factor_weights[factor]
        
        # Select top N stocks
        factor_df = factor_df.dropna(subset=['COMPOSITE_SCORE'])
        top_stocks = factor_df.nlargest(top_n, 'COMPOSITE_SCORE')
        
        # Get current prices
        top_stocks = pd.merge(top_stocks, latest_prices, on='SYMBOL')
        
        # Liquidate current holdings
        for symbol, quantity in holdings.items():
            if symbol in latest_prices['SYMBOL'].values:
                price = latest_prices[latest_prices['SYMBOL'] == symbol]['CLOSE_PRICE'].values[0]
                cash += quantity * price
        
        # Reset holdings
        holdings = {}
        
        # Allocate equally to top stocks
        if len(top_stocks) > 0:
            amount_per_stock = cash / len(top_stocks)
            for _, row in top_stocks.iterrows():
                quantity = amount_per_stock / row['CLOSE_PRICE']
                holdings[row['SYMBOL']] = quantity
            cash = 0  # All cash invested
        
        # Calculate daily portfolio values for each day between rebalance dates
        interim_dates = [d for d in all_dates if current_date <= d < next_date]
        
        for interim_date in interim_dates:
            day_prices = df[df['DATE'] == interim_date][['SYMBOL', 'CLOSE_PRICE']]
            day_portfolio_value = cash
            
            for symbol, quantity in holdings.items():
                if symbol in day_prices['SYMBOL'].values:
                    price = day_prices[day_prices['SYMBOL'] == symbol]['CLOSE_PRICE'].values[0]
                    day_portfolio_value += quantity * price
            
            # Store daily portfolio value (for drawdown calculation)
            normalized_day_value = day_portfolio_value / absolute_portfolio_values[0][1]
            daily_portfolio_values.append((interim_date, normalized_day_value))
        
        # Calculate portfolio value at next rebalance date
        next_prices = df[df['DATE'] == next_date][['SYMBOL', 'CLOSE_PRICE']]
        portfolio_value = cash
        
        for symbol, quantity in holdings.items():
            if symbol in next_prices['SYMBOL'].values:
                price = next_prices[next_prices['SYMBOL'] == symbol]['CLOSE_PRICE'].values[0]
                portfolio_value += quantity * price
        
        # Store absolute value
        absolute_portfolio_values.append((next_date, portfolio_value))
        holdings_history.append((next_date, holdings.copy()))
        
        # Calculate normalized values (relative to starting point)
        normalized_portfolio_value = portfolio_value / absolute_portfolio_values[0][1]
        portfolio_values.append((next_date, normalized_portfolio_value))
        
        # Get Nifty 50 value for comparison
        try:
            # Try to get exact match first
            next_nifty_data = nifty_data[nifty_data['DATE'] == next_date]
            if len(next_nifty_data) > 0:
                next_nifty_price = next_nifty_data.iloc[0]['CLOSE_PRICE']
            else:
                # If no exact match, get the closest previous date
                next_nifty_data = nifty_data[nifty_data['DATE'] <= next_date]
                if len(next_nifty_data) > 0:
                    next_nifty_price = next_nifty_data.iloc[-1]['CLOSE_PRICE']
                else:
                    raise IndexError("No Nifty data found")
                    
            # Calculate normalized value
            normalized_nifty = next_nifty_price / initial_nifty
            nifty_values.append((next_date, normalized_nifty))
        except (IndexError, KeyError):
            # If Nifty data is missing for this date, use last known value
            if nifty_values:
                nifty_values.append((next_date, nifty_values[-1][1]))
            else:
                nifty_values.append((next_date, 1.0))
    
    # Clear the progress bar
    progress_bar.empty()
    
    # Prepare results
    portfolio_df = pd.DataFrame(portfolio_values, columns=['DATE', 'PORTFOLIO_VALUE'])
    nifty_df = pd.DataFrame(nifty_values, columns=['DATE', 'MARKET_VALUE'])
    
    # Create drawdown dataframe
    drawdown_df = pd.DataFrame(daily_portfolio_values, columns=['DATE', 'PORTFOLIO_VALUE'])
    
    # Merge portfolio and market data
    results = pd.merge(portfolio_df, nifty_df, on='DATE')
    
    # Calculate performance metrics using normalized values
    portfolio_returns = results['PORTFOLIO_VALUE'].pct_change()
    market_returns = results['MARKET_VALUE'].pct_change()
    
    # Annualized return
    days = (results['DATE'].iloc[-1] - results['DATE'].iloc[0]).days
    portfolio_cagr = (results['PORTFOLIO_VALUE'].iloc[-1] / results['PORTFOLIO_VALUE'].iloc[0]) ** (365/days) - 1
    market_cagr = (results['MARKET_VALUE'].iloc[-1] / results['MARKET_VALUE'].iloc[0]) ** (365/days) - 1
    
    # Volatility
    portfolio_vol = portfolio_returns.std() * np.sqrt(252)
    market_vol = market_returns.std() * np.sqrt(252)
    
    # Sharpe Ratio (assuming risk-free rate of 5%)
    rf_daily = 0.05 / 252
    portfolio_sharpe = (portfolio_returns.mean() - rf_daily) / portfolio_returns.std() * np.sqrt(252)
    market_sharpe = (market_returns.mean() - rf_daily) / market_returns.std() * np.sqrt(252)
    
    # Sortino Ratio (downside risk only)
    negative_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0001
    portfolio_sortino = (portfolio_returns.mean() - rf_daily) / downside_deviation * np.sqrt(252)
    
    # Market Sortino Ratio
    market_negative_returns = market_returns[market_returns < 0]
    market_downside_deviation = market_negative_returns.std() * np.sqrt(252) if len(market_negative_returns) > 0 else 0.0001
    market_sortino = (market_returns.mean() - rf_daily) / market_downside_deviation * np.sqrt(252)
    
    # Maximum drawdown calculation
    drawdown_df['ROLLING_MAX'] = drawdown_df['PORTFOLIO_VALUE'].cummax()
    drawdown_df['DRAWDOWN'] = (drawdown_df['PORTFOLIO_VALUE'] - drawdown_df['ROLLING_MAX']) / drawdown_df['ROLLING_MAX']
    max_drawdown = drawdown_df['DRAWDOWN'].min()
    
    # Drawdown duration
    is_drawdown = drawdown_df['DRAWDOWN'] < 0
    drawdown_starts = is_drawdown & ~is_drawdown.shift(1, fill_value=False)
    drawdown_ends = ~is_drawdown & is_drawdown.shift(1, fill_value=False)
    
    # Calculate drawdown durations
    drawdown_durations = []
    if sum(drawdown_starts) > 0 and sum(drawdown_ends) > 0:
        start_indices = drawdown_df.index[drawdown_starts].tolist()
        end_indices = drawdown_df.index[drawdown_ends].tolist()
        
        # Ensure we have matching start and end points
        if len(end_indices) < len(start_indices):
            # If last drawdown hasn't ended, use last date
            end_indices.append(len(drawdown_df) - 1)
        
        # Calculate durations
        for i in range(min(len(start_indices), len(end_indices))):
            start_idx = start_indices[i]
            end_idx = end_indices[i]
            if end_idx > start_idx:  # valid drawdown period
                start_date = drawdown_df.iloc[start_idx]['DATE']
                end_date = drawdown_df.iloc[end_idx]['DATE']
                duration_days = (end_date - start_date).days
                drawdown_durations.append(duration_days)
    
    # Max drawdown duration
    max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0
    
    metrics = {
        'CAGR': portfolio_cagr,
        'Volatility': portfolio_vol,
        'Sharpe': portfolio_sharpe,
        'Sortino': portfolio_sortino,
        'Max Drawdown': max_drawdown,
        'Max Drawdown Duration (days)': max_drawdown_duration,
        'Market CAGR': market_cagr,
        'Market Volatility': market_vol,
        'Market Sharpe': market_sharpe,
        'Market Sortino': market_sortino
    }
    
    # Get current holdings
    latest_holdings = holdings_history[-1][1]
    latest_prices = df[df['DATE'] == df['DATE'].max()][['SYMBOL', 'CLOSE_PRICE']]
    
    holdings_df = pd.DataFrame({
        'SYMBOL': list(latest_holdings.keys()),
        'QUANTITY': list(latest_holdings.values())
    })
    
    holdings_df = pd.merge(holdings_df, latest_prices, on='SYMBOL')
    holdings_df['VALUE'] = holdings_df['QUANTITY'] * holdings_df['CLOSE_PRICE']
    holdings_df['WEIGHT'] = holdings_df['VALUE'] / holdings_df['VALUE'].sum()
    
    return results, metrics, holdings_df, drawdown_df

# Initialize session state to store our data and calculation results
if 'factor_weights' not in st.session_state:
    st.session_state.factor_weights = {
        'value': 0.25,
        'momentum': 0.25,
        'volatility': 0.25,
        'quality': 0.25
    }
    
if 'active_factors' not in st.session_state:
    st.session_state.active_factors = ['value', 'momentum', 'volatility', 'quality']
    
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
    
if 'results' not in st.session_state:
    st.session_state.results = None
    
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
    
if 'holdings' not in st.session_state:
    st.session_state.holdings = None
    
if 'factor_data' not in st.session_state:
    st.session_state.factor_data = None

if 'drawdown_data' not in st.session_state:
    st.session_state.drawdown_data = None

if 'recently_changed_factor' not in st.session_state:
    st.session_state.recently_changed_factor = None

# Function to normalize weights while preserving the most recently changed factor
def normalize_weights():
    active_factors = st.session_state.active_factors
    if not active_factors:
        return
    
    # Edge case: prevent division by zero if all weights are 0
    if sum(st.session_state.factor_weights[f] for f in active_factors) == 0:
        # Set equal weights
        for factor in active_factors:
            st.session_state.factor_weights[factor] = 1.0 / len(active_factors)
        return
    
    # Set inactive factors to 0
    for factor in ['value', 'momentum', 'volatility', 'quality']:
        if factor not in active_factors:
            st.session_state.factor_weights[factor] = 0
    
    # If we have a recently changed factor, preserve its weight
    if 'recently_changed_factor' in st.session_state and st.session_state.recently_changed_factor in active_factors:
        changed_factor = st.session_state.recently_changed_factor
        changed_weight = st.session_state.factor_weights[changed_factor]
        
        # Calculate how much weight remains for other factors
        remaining_weight = 1.0 - changed_weight
        
        # Get the current weights of other active factors
        other_active_factors = [f for f in active_factors if f != changed_factor]
        other_weights_sum = sum(st.session_state.factor_weights[f] for f in other_active_factors)
        
        # Distribute the remaining weight proportionally among other active factors
        if other_weights_sum > 0 and other_active_factors:  # Prevent division by zero
            for factor in other_active_factors:
                st.session_state.factor_weights[factor] = (st.session_state.factor_weights[factor] / other_weights_sum) * remaining_weight
        elif other_active_factors:  # If all other weights are 0, distribute equally
            equal_weight = remaining_weight / len(other_active_factors)
            for factor in other_active_factors:
                st.session_state.factor_weights[factor] = equal_weight
    else:
        # If no recently changed factor, normalize all weights
        total_weight = sum(st.session_state.factor_weights[f] for f in active_factors)
        
        # Normalize weights to sum to 1
        if total_weight > 0:
            for factor in active_factors:
                st.session_state.factor_weights[factor] = st.session_state.factor_weights[factor] / total_weight

# Define a callback function to update weights when a slider changes
def on_slider_change(factor_name):
    """Updates all weights when one slider changes"""
    # Get the new value for the changed factor
    new_value = st.session_state[f'{factor_name}_slider']
    
    # Update the factor weight
    st.session_state.factor_weights[factor_name] = new_value
    
    # Mark this factor as recently changed
    st.session_state.recently_changed_factor = factor_name
    
    # Normalize weights while preserving this factor's new weight
    normalize_weights()

# Reset weights to equal function
def reset_weights():
    """Reset all weights to equal"""
    active_factors = st.session_state.active_factors
    if not active_factors:
        return
    
    equal_weight = 1.0 / len(active_factors)
    for factor in active_factors:
        st.session_state.factor_weights[factor] = equal_weight
    
    # Clear recently changed factor
    st.session_state.recently_changed_factor = None

st.sidebar.write("Created by:")
linkedin_url = "https://www.linkedin.com/in/jeevanba273/"
st.sidebar.markdown(
    f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: black;">'
    f'<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">'
    f'JEEVAN B A</a>',
    unsafe_allow_html=True
)

# Sidebar for factor selection and weights
st.sidebar.header("Factor Selection")

# Checkboxes for enabling/disabling factors
value_enabled = st.sidebar.checkbox("Value Factor", value=True, key='value_enabled')
momentum_enabled = st.sidebar.checkbox("Momentum Factor", value=True, key='momentum_enabled')
volatility_enabled = st.sidebar.checkbox("Low Volatility Factor", value=True, key='volatility_enabled')
quality_enabled = st.sidebar.checkbox("Quality Factor", value=True, key='quality_enabled')

# Add factor descriptions as help text
st.sidebar.markdown("""
<div style="font-size:0.8em; color:gray; margin-top:10px;">
<strong>Factor Descriptions:</strong><br>
• Value: Identifies undervalued stocks based on price-to-volume<br>
• Momentum: Captures stocks with positive price trends<br>
• Low Volatility: Favors stocks with lower price fluctuations<br>
• Quality: Focuses on stocks with stable trading and stronger investor holdings
</div>
""", unsafe_allow_html=True)

# Update active factors list based on checkboxes
active_factors = []
if value_enabled:
    active_factors.append('value')
if momentum_enabled:
    active_factors.append('momentum')
if volatility_enabled:
    active_factors.append('volatility')
if quality_enabled:
    active_factors.append('quality')

# Store active factors in session state
st.session_state.active_factors = active_factors

# Sidebar for factor weights
st.sidebar.header("Factor Weights")

# Initialize sliders for active factors
st.sidebar.subheader("Adjust weights for active factors")

# Initialize sliders with current weight values and proper callbacks
if 'value' in active_factors:
    value_weight = st.sidebar.slider(
        "Value Factor Weight", 
        0.0, 1.0, 
        st.session_state.factor_weights['value'],  # This initializes the slider
        0.05,
        key='value_slider',
        on_change=on_slider_change,
        args=('value',)
    )

if 'momentum' in active_factors:
    momentum_weight = st.sidebar.slider(
        "Momentum Factor Weight", 
        0.0, 1.0, 
        st.session_state.factor_weights['momentum'], 
        0.05,
        key='momentum_slider',
        on_change=on_slider_change,
        args=('momentum',)
    )

if 'volatility' in active_factors:
    vol_weight = st.sidebar.slider(
        "Low Volatility Factor Weight", 
        0.0, 1.0, 
        st.session_state.factor_weights['volatility'], 
        0.05,
        key='volatility_slider',
        on_change=on_slider_change,
        args=('volatility',)
    )

if 'quality' in active_factors:
    quality_weight = st.sidebar.slider(
        "Quality Factor Weight", 
        0.0, 1.0, 
        st.session_state.factor_weights['quality'], 
        0.05,
        key='quality_slider',
        on_change=on_slider_change,
        args=('quality',)
    )

# Reset weights button - positioned below sliders with custom styling
_, col2, _ = st.sidebar.columns([0.75, 3.5, 0.75])  # Wider middle column
with col2:
    if st.button("Reset to\nEqual Weights", on_click=reset_weights, key="reset_weights_btn"):
        st.rerun()

# Normalize weights when factors are toggled
if st.session_state.active_factors != active_factors:
    # Store the new active factors
    st.session_state.active_factors = active_factors
    # Clear the recently changed factor when toggling
    st.session_state.recently_changed_factor = None
    # Normalize weights
    normalize_weights()

# Display current weights
st.sidebar.subheader("Current Weights")
for factor in ['value', 'momentum', 'volatility', 'quality']:
    if factor in active_factors:
        st.sidebar.write(f"{factor.capitalize()}: {st.session_state.factor_weights[factor]:.2f}")

# Portfolio settings
st.sidebar.header("Portfolio Settings")
num_stocks = st.sidebar.slider("Number of Stocks", 10, 100, 30)
rebalance_freq = st.sidebar.selectbox(
    "Rebalancing Frequency",
    options=["monthly", "quarterly", "annual"],
    index=0
)

# Date range selection for backtesting
st.sidebar.header("Backtest Period")

# Load data
data = load_data()

# Set min and max dates from the data
min_date = data['DATE'].min().date()
max_date = data['DATE'].max().date()

# Default to 1 year backtest
default_start_date = max_date - timedelta(days=365)
if default_start_date < min_date:
    default_start_date = min_date

# Add a spacer before the date inputs to give more room
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Expanded container for dates to ensure better display
with st.sidebar.container():
    # Date inputs with custom key to force refresh
    start_date = st.date_input(
        "Start Date",
        value=default_start_date,
        min_value=min_date,
        max_value=max_date,
        key="start_date_input"
    )
    
    end_date = st.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="end_date_input"
    )

# Convert to datetime for processing
start_datetime = pd.Timestamp(start_date)
end_datetime = pd.Timestamp(end_date)

# Ensure end date is after start date
if start_datetime >= end_datetime:
    st.sidebar.error("End date must be after start date")
    end_datetime = start_datetime + pd.Timedelta(days=1)

# Add option to reduce data size for faster processing
st.sidebar.header("Performance Options")
use_reduced_dataset = st.sidebar.checkbox("Use reduced dataset (faster)", value=True, key="use_reduced_dataset")

if use_reduced_dataset:
    st.sidebar.info("Using a reduced dataset for faster processing. Uncheck for full analysis.")

# Add Run Analysis button to sidebar
st.sidebar.header("Run Analysis")
run_button = st.sidebar.button("Run Analysis", on_click=lambda: setattr(st.session_state, 'run_analysis', True))

# Only run analysis if the button is clicked
if st.session_state.run_analysis:
    # Check if any factors are active
    if active_factors:
        try:
            # Apply dataset reduction if selected
            working_data = data
            if use_reduced_dataset:
                # Filter to relevant dates to reduce processing load
                date_mask = (working_data['DATE'] >= start_datetime - pd.Timedelta(days=400)) & (working_data['DATE'] <= end_datetime)
                working_data = working_data[date_mask].copy()
                st.info("Using reduced dataset for faster processing...")
            
            # Calculate factor model
            st.write("Calculating factor model...")
            factor_df = create_factor_model(working_data, active_factors)
            st.session_state.factor_data = factor_df
            
            # Run backtest
            st.write("Running backtest... This may take a few minutes.")
            results, metrics, holdings, drawdown_df = backtest_factor_portfolio(
                working_data, 
                st.session_state.factor_weights,
                active_factors,
                top_n=num_stocks, 
                start_date=start_datetime,
                end_date=end_datetime,
                rebalance_freq=rebalance_freq
            )
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.metrics = metrics
            st.session_state.holdings = holdings
            st.session_state.drawdown_data = drawdown_df
        
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
    else:
        st.error("Please select at least one factor to build the portfolio.")
    
    # Reset flag after analysis is complete
    st.session_state.run_analysis = False

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Factor Analysis", "Portfolio Backtest", "Current Portfolio"])

with tab1:
    st.header("Factor Analysis")
    
    if st.session_state.factor_data is not None and active_factors:
        factor_df = st.session_state.factor_data
        
        # Show factor correlations if at least 2 factors are active
        active_factor_columns = [f"{factor.upper()}_SCORE" for factor in active_factors]
        if len(active_factor_columns) >= 2:
            st.subheader("Factor Correlations")
            factor_corr = factor_df[active_factor_columns].corr()
            
            # Create readable labels for the correlation matrix
            factor_labels = [factor.capitalize() for factor in active_factors]
            
            fig_corr = px.imshow(
                factor_corr, 
                text_auto=True, 
                color_continuous_scale='RdBu_r',
                labels=dict(x="Factor", y="Factor", color="Correlation"),
                x=factor_labels,
                y=factor_labels
            )
            fig_corr.update_layout(height=400, width=500)
            st.plotly_chart(fig_corr)
        
        # Display top stocks for each active factor
        st.subheader("Top 10 Stocks by Factor")
        
        # Create columns dynamically based on number of active factors
        if active_factors:
            num_cols = min(2, len(active_factors))
            factor_cols = st.columns(num_cols)
            
            for i, factor in enumerate(active_factors):
                col_idx = i % num_cols
                with factor_cols[col_idx]:
                    factor_col = f"{factor.upper()}_SCORE"
                    if factor_col in factor_df.columns:
                        st.write(f"**{factor.capitalize()} Factor**")
                        st.dataframe(factor_df.nlargest(10, factor_col)[['SYMBOL', factor_col]])
    else:
        if not active_factors:
            st.info("Please select at least one factor and click 'Run Analysis'.")
        else:
            st.info("Please click 'Run Analysis' in the sidebar to calculate factors.")

with tab2:
    st.header("Portfolio Backtest")
    
    if st.session_state.results is not None and st.session_state.metrics is not None:
        results = st.session_state.results
        metrics = st.session_state.metrics
        drawdown_df = st.session_state.drawdown_data
        
        # Plot performance
        st.subheader("Portfolio Performance vs Market")
        
        # Create the plotly figure with vertical cursor
        fig = go.Figure()
        
        # Add portfolio trace
        fig.add_trace(go.Scatter(
            x=results['DATE'],
            y=results['PORTFOLIO_VALUE'],
            mode='lines',
            name='Factor Portfolio'
        ))
        
        # Add market trace
        fig.add_trace(go.Scatter(
            x=results['DATE'],
            y=results['MARKET_VALUE'],
            mode='lines',
            name='Market'
        ))
        
        # Configure layout with hover mode
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Value (Normalized to 1)",
            height=500,
            hovermode='x unified',
            hoverdistance=100,
            spikedistance=1000,
        )
        
        # Format y-axis as percentage
        fig.update_yaxes(
            tickformat=".2f",
            hoverformat=".2f"
        )
        
        # Add spikelines (vertical cursor line)
        fig.update_xaxes(
            showspikes=True,
            spikethickness=1,
            spikecolor="gray",
            spikemode="across",
            spikedash="solid"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot drawdown chart
        st.subheader("Portfolio Drawdown")
        
        fig_drawdown = go.Figure()
        
        # Add drawdown trace
        fig_drawdown.add_trace(go.Scatter(
            x=drawdown_df['DATE'],
            y=drawdown_df['DRAWDOWN'] * 100,  # Convert to percentage
            mode='lines',
            name='Drawdown',
            line=dict(color='red')
        ))
        
        # Configure layout
        fig_drawdown.update_layout(
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=300,
            hovermode='x unified',
            hoverdistance=100,
            spikedistance=1000,
        )
        
        # Format y-axis and add reference line at 0
        fig_drawdown.update_yaxes(
            tickformat=".1f",
            hoverformat=".2f"
        )
        
        # Add reference line at 0
        fig_drawdown.add_shape(
            type="line",
            x0=drawdown_df['DATE'].min(),
            y0=0,
            x1=drawdown_df['DATE'].max(),
            y1=0,
            line=dict(
                color="gray",
                width=1,
                dash="dash",
            )
        )
        
        st.plotly_chart(fig_drawdown, use_container_width=True)
        
        # Display metrics
        st.subheader("Performance Metrics")
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.markdown(f"**Portfolio**")
            st.markdown(f"{tooltip('CAGR:', 'Compound Annual Growth Rate - The annual rate of return that would be required for an investment to grow from its beginning value to its ending value.')} {metrics['CAGR']:.2%}", unsafe_allow_html=True)
            st.markdown(f"{tooltip('Volatility:', 'Annualized standard deviation of returns, measuring the dispersion of returns around the mean.')} {metrics['Volatility']:.2%}", unsafe_allow_html=True)
            st.markdown(f"{tooltip('Sharpe Ratio:', 'Risk-adjusted return, calculated as excess return over risk-free rate divided by standard deviation of returns.')} {metrics['Sharpe']:.2f}", unsafe_allow_html=True)
            st.markdown(f"{tooltip('Sortino Ratio:', 'Similar to Sharpe ratio but only penalizes downside volatility, measuring return relative to harmful volatility.')} {metrics['Sortino']:.2f}", unsafe_allow_html=True)
            st.markdown(f"{tooltip('Max Drawdown:', 'Largest percentage drop from peak to trough during the investment period.')} {metrics['Max Drawdown']:.2%}", unsafe_allow_html=True)
            st.markdown(f"{tooltip('Max Drawdown Duration:', 'Longest period (in days) from peak to recovery to the same level.')} {metrics['Max Drawdown Duration (days)']:.0f} days", unsafe_allow_html=True)
        
        with metrics_col2:
            st.markdown(f"**Market Benchmark**")
            st.markdown(f"CAGR: {metrics['Market CAGR']:.2%}")
            st.markdown(f"Volatility: {metrics['Market Volatility']:.2%}")
            st.markdown(f"Sharpe Ratio: {metrics['Market Sharpe']:.2f}")
            st.markdown(f"Sortino Ratio: {metrics['Market Sortino']:.2f}")
    else:
        if not active_factors:
            st.info("Please select at least one factor and click 'Run Analysis'.")
        else:
            st.info("Please click 'Run Analysis' in the sidebar to view backtest results.")

with tab3:
    st.header("Current Portfolio Composition")
    
    if st.session_state.holdings is not None and st.session_state.factor_data is not None:
        current_holdings = st.session_state.holdings
        factor_df = st.session_state.factor_data
        
        # Display current holdings
        st.subheader("Current Holdings")
        st.dataframe(current_holdings[['SYMBOL', 'QUANTITY', 'CLOSE_PRICE', 'VALUE', 'WEIGHT']])
        
        # Pie chart of portfolio allocation
        fig_pie = px.pie(
            current_holdings,
            values='WEIGHT',
            names='SYMBOL',
            title='Portfolio Allocation'
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Factor exposure of current portfolio
        st.subheader("Factor Exposure")
        
        # Dynamically merge only the active factor score columns
        active_factor_columns = [f"{factor.upper()}_SCORE" for factor in st.session_state.active_factors]
        factor_exposure = pd.merge(
            current_holdings[['SYMBOL', 'WEIGHT']], 
            factor_df[['SYMBOL'] + active_factor_columns], 
            on='SYMBOL'
        )
        
        # Calculate weighted factor exposures dynamically
        weighted_exposures = {}
        for factor in st.session_state.active_factors:
            col = f"{factor.upper()}_SCORE"
            factor_name = "Low Volatility" if factor == "volatility" else factor.capitalize()
            weighted_exposures[factor_name] = (factor_exposure[col] * factor_exposure['WEIGHT']).sum()
        
        exposure_df = pd.DataFrame({
            'Factor': list(weighted_exposures.keys()),
            'Exposure': list(weighted_exposures.values())
        })
        
        fig_exposure = px.bar(
            exposure_df,
            x='Factor',
            y='Exposure',
            title='Portfolio Factor Exposure'
        )
        st.plotly_chart(fig_exposure, use_container_width=True)
        
        # Add export functionality
        st.header("Export Portfolio")
        csv = current_holdings.to_csv(index=False)
        st.download_button(
            label="Download Current Portfolio",
            data=csv,
            file_name="smart_beta_portfolio.csv",
            mime="text/csv"
        )
    else:
        st.info("Please click 'Run Analysis' in the sidebar to view portfolio composition.")