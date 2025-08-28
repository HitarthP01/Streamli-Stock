"""
Configuration Module
Contains all configuration settings and constants.
"""

# Database configuration
DATABASE_CONFIG = {
    'default_db_name': "backup_v1.db",
    'timeout': 30.0,
    'minimum_historical_days': 60
}

# API configuration  
API_CONFIG = {
    'yahoo_finance_url': 'https://finance.yahoo.com/research-hub/screener/most_actives/',
    'request_timeout': 10,
    'headers': {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
}

# Technical indicators configuration
TECHNICAL_CONFIG = {
    'sma_short_period': 20,
    'sma_long_period': 50,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30
}

# Market cap thresholds
MARKET_CAP_CONFIG = {
    'high_cap_threshold': 100_000_000_000,  # 100 billion
    'data_density_threshold': 0.7  # 70% data completeness
}

# Streamlit page configuration
STREAMLIT_CONFIG = {
    'page_title': "Advanced Stock Market Dashboard",
    'page_icon': "📈",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Custom CSS styles
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stock-positive {
        color: #00ff00;
        font-weight: bold;
    }
    .stock-negative {
        color: #ff4444;
        font-weight: bold;
    }
    .data-table {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
</style>
"""
