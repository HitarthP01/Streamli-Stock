"""
Stock Visualizer Module
Handles all chart creation and visualization operations.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class StockVisualizer:
    @staticmethod
    def create_advanced_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create advanced candlestick chart with technical indicators"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2],
            vertical_spacing=0.05,
            subplot_titles=('Price & Technical Indicators', 'Volume', 'RSI')
        )
        
        # Candlestick chart
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(
                go.Candlestick(
                    x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='OHLC',
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff4444'
                ),
                row=1, col=1
            )
        
        # Moving averages
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], y=df['sma_20'],
                    mode='lines', name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], y=df['sma_50'],
                    mode='lines', name='SMA 50',
                    line=dict(color='purple', width=1)
                ),
                row=1, col=1
            )
        
        # Volume
        if 'volume' in df.columns:
            colors = ['green' if close >= open else 'red' 
                     for close, open in zip(df['close'], df['open'])]
            fig.add_trace(
                go.Bar(
                    x=df['date'], y=df['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], y=df['rsi'],
                    mode='lines', name='RSI',
                    line=dict(color='blue', width=2)
                ),
                row=3, col=1
            )
            
            # RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol.upper()} - Advanced Stock Chart",
            xaxis_rangeslider_visible=False,
            height=800,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        
        return fig
