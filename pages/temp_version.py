import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Crypto Heatmap Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitCryptoAnalyzer:
    def __init__(self):
        """Initialize with popular cryptocurrency symbols"""
        self.crypto_symbols = {
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD',
            'Cardano': 'ADA-USD',
            'Solana': 'SOL-USD',
            'Polygon': 'MATIC-USD',
            'Chainlink': 'LINK-USD',
            'Litecoin': 'LTC-USD',
            'Polkadot': 'DOT-USD',
            'Avalanche': 'AVAX-USD',
            'Dogecoin': 'DOGE-USD',
            'Shiba Inu': 'SHIB-USD',
            'Cosmos': 'ATOM-USD',
            'Algorand': 'ALGO-USD',
            'VeChain': 'VET-USD',
            'Tezos': 'XTZ-USD'
        }
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_crypto_data(_self, symbol, start_year, end_year):
        """
        Fetch cryptocurrency data with caching for better performance
        """
        try:
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year + 1}-01-01"
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if data.empty:
                return None, f"No data found for {symbol}"
            
            return data, None
            
        except Exception as e:
            return None, f"Error fetching data: {str(e)}"
    
    def calculate_monthly_returns(self, data):
        """Calculate monthly percentage returns"""
        try:
            # Resample to monthly (last trading day of each month)
            monthly_prices = data['Close'].resample('M').last()
            
            # Calculate monthly percentage change
            monthly_returns = monthly_prices.pct_change() * 100
            
            # Create comprehensive DataFrame
            monthly_df = pd.DataFrame({
                'date': monthly_prices.index,
                'price': monthly_prices.values,
                'monthly_return': monthly_returns.values,
                'year': monthly_prices.index.year,
                'month': monthly_prices.index.month,
                'month_name': monthly_prices.index.strftime('%b')
            })
            
            # Remove first month (NaN return)
            monthly_df = monthly_df.dropna()
            
            return monthly_df, None
            
        except Exception as e:
            return None, f"Error calculating returns: {str(e)}"
    
    def create_heatmap_data(self, monthly_df):
        """Create pivot table for heatmap"""
        try:
            heatmap_data = monthly_df.pivot_table(
                values='monthly_return',
                index='year',
                columns='month',
                aggfunc='first'
            )
            
            # Add proper month labels
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Only rename columns that exist in the data
            existing_months = heatmap_data.columns
            heatmap_data.columns = [month_labels[i-1] for i in existing_months]
            
            return heatmap_data, None
            
        except Exception as e:
            return None, f"Error creating heatmap data: {str(e)}"
    
    def calculate_summary_stats(self, monthly_df):
        """Calculate comprehensive summary statistics"""
        returns = monthly_df['monthly_return']
        
        stats = {
            'Total Months': len(returns),
            'Average Monthly Return (%)': returns.mean(),
            'Median Monthly Return (%)': returns.median(),
            'Standard Deviation (%)': returns.std(),
            'Best Month (%)': returns.max(),
            'Worst Month (%)': returns.min(),
            'Positive Months': (returns > 0).sum(),
            'Negative Months': (returns < 0).sum(),
            'Win Rate (%)': (returns > 0).mean() * 100,
            'Sharpe Ratio': returns.mean() / returns.std() if returns.std() != 0 else 0
        }
        
        return stats

def create_plotly_heatmap(heatmap_data, crypto_name):
    """Create interactive Plotly heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        # colorscale='RdYlGn',
        colorscale=[
            [0, 'red'],     # Negative returns
            [0.5, 'red'],   # Slightly negative returns
            [0.5, 'green'], # Slightly positive returns
            [1, 'green']    # Positive returns
        ],
        zmid=0,
        text=np.round(heatmap_data.values, 1),
        texttemplate="%{text}%",
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': f'{crypto_name} Monthly Returns Heatmap',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Arial, sans-serif'}
        },
        xaxis_title="Month",
        yaxis_title="Year",
        height=max(400, len(heatmap_data) * 25),
        width=800,
        font=dict(size=12)
    )
    
    return fig

def main():
    # Initialize analyzer
    analyzer = StreamlitCryptoAnalyzer()
    
    # App header
    st.title("📈 Cryptocurrency Monthly Returns Heatmap")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Configuration")
    
    # Cryptocurrency selection
    selected_crypto = st.sidebar.selectbox(
        "Select Cryptocurrency:",
        options=list(analyzer.crypto_symbols.keys()),
        index=0
    )
    
    # Date range selection
    current_year = datetime.now().year
    start_year = st.sidebar.slider(
        "Start Year:",
        min_value=2010,
        max_value=current_year - 1,
        value=current_year - 15,
        step=1
    )
    
    end_year = st.sidebar.slider(
        "End Year:",
        min_value=start_year + 1,
        max_value=current_year,
        value=current_year,
        step=1
    )
    
    # Analysis button
    analyze_button = st.sidebar.button("🚀 Generate Heatmap", type="primary")
    
    # Main content area
    if analyze_button:
        symbol = analyzer.crypto_symbols[selected_crypto]
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"Fetching {selected_crypto} data...")
        progress_bar.progress(25)
        
        # Fetch data
        raw_data, error = analyzer.fetch_crypto_data(symbol, start_year, end_year)
        
        if error:
            st.error(f"❌ {error}")
            return
        
        status_text.text("Calculating monthly returns...")
        progress_bar.progress(50)
        
        # Calculate monthly returns
        monthly_df, error = analyzer.calculate_monthly_returns(raw_data)
        
        if error:
            st.error(f"❌ {error}")
            return
        
        status_text.text("Creating heatmap...")
        progress_bar.progress(75)
        
        # Create heatmap data
        heatmap_data, error = analyzer.create_heatmap_data(monthly_df)
        
        if error:
            st.error(f"❌ {error}")
            return
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Display results
        st.markdown("---")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        stats = analyzer.calculate_summary_stats(monthly_df)
        
        with col1:
            st.metric(
                label="Avg Monthly Return",
                value=f"{stats['Average Monthly Return (%)']:.2f}%"
            )
        
        with col2:
            st.metric(
                label="Win Rate",
                value=f"{stats['Win Rate (%)']:.1f}%"
            )
        
        with col3:
            st.metric(
                label="Best Month",
                value=f"{stats['Best Month (%)']:.1f}%"
            )
        
        with col4:
            st.metric(
                label="Worst Month",
                value=f"{stats['Worst Month (%)']:.1f}%"
            )
        
        # Main heatmap
        st.markdown("### 🔥 Monthly Returns Heatmap")
        fig = create_plotly_heatmap(heatmap_data, selected_crypto)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Summary Statistics")
            stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
            
            # Format the display
            for key, value in stats.items():
                if isinstance(value, float):
                    if '%' in key:
                        st.write(f"**{key}:** {value:.2f}%")
                    else:
                        st.write(f"**{key}:** {value:.2f}")
                else:
                    st.write(f"**{key}:** {value}")
        
        with col2:
            st.markdown("### 📈 Monthly Distribution")
            
            # Create histogram of monthly returns
            fig_hist = px.histogram(
                data_frame = monthly_df,
                x='monthly_return',
                nbins=30,
                title='Distribution of Monthly Returns',
                labels = {'monthly_return': 'Monthly Return (%)', 'count': 'Frequency'},
                color_discrete_sequence=['#2ecc71'],
                opacity=0.75,
                marginal='box'  # Add a box plot on top
            )

            fig_hist.add_vline(x=0, line_dash="dash", line_color="red", 
                              annotation_text="Break-even")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Raw Data"):
            st.dataframe(
                monthly_df[['date', 'price', 'monthly_return']].round(2),
                use_container_width=True
            )
        
        # Download option
        csv = monthly_df.to_csv(index=False)
        st.download_button(
            label="💾 Download Data as CSV",
            data=csv,
            file_name=f"{selected_crypto}_monthly_returns_{start_year}_{end_year}.csv",
            mime="text/csv"
        )
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
    
    else:
        # Initial state - show instructions
        st.markdown("""
        ### 🎯 How to Use This Tool:
        
        1. **Select a Cryptocurrency** from the sidebar dropdown
        2. **Choose your date range** using the year sliders
        3. **Click "Generate Heatmap"** to analyze the data
        
        ### 🔍 What You'll Get:
        - **Interactive Heatmap** showing monthly returns by year
        - **Summary Statistics** with key performance metrics
        - **Return Distribution** histogram
        - **Downloadable Data** in CSV format
        
        ### 💡 Color Coding:
        - 🟢 **Green**: Positive returns (good months)
        - 🟡 **Yellow**: Near zero returns
        - 🔴 **Red**: Negative returns (bad months)
        """)
        
        # Show available cryptocurrencies
        st.markdown("### 💰 Available Cryptocurrencies:")
        crypto_cols = st.columns(3)
        crypto_list = list(analyzer.crypto_symbols.keys())
        
        for i, crypto in enumerate(crypto_list):
            with crypto_cols[i % 3]:
                st.write(f"• {crypto}")

if __name__ == "__main__":
    main()