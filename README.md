# 🎯 QuantitativeTradingAI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-18%20Models-success.svg)](https://github.com/yourusername/QuantitativeTradingAI)
[![Trading Agents](https://img.shields.io/badge/Trading%20Agents-23%20Models-success.svg)](https://github.com/yourusername/QuantitativeTradingAI)

A comprehensive quantitative trading and stock prediction platform featuring state-of-the-art machine learning models, reinforcement learning agents, and real-time trading capabilities.

## 🌟 Features

### 🤖 **Deep Learning Models (18 Models)**
- **LSTM Variants**: Standard, Bidirectional, 2-Path, Seq2Seq, VAE
- **GRU Variants**: Standard, Bidirectional, 2-Path, Seq2Seq, VAE  
- **Transformer**: Attention-is-all-you-Need architecture
- **CNN**: Convolutional Neural Networks for time series
- **Ensemble Methods**: Stacking, Autoencoder-based approaches

### 🎮 **Trading Agents (23 Models)**
- **Reinforcement Learning**: Q-Learning, Double Q-Learning, Duel Q-Learning
- **Actor-Critic**: Standard, Duel, Recurrent variants
- **Evolution Strategies**: Neuro-evolution, Novelty Search
- **Traditional**: Turtle Trading, Moving Average, Signal Rolling
- **Advanced**: Curiosity-driven, ABCD Strategy

### 📊 **Real-time Trading**
- **Flask API**: RESTful endpoints for live trading
- **Web Interface**: Interactive dashboard
- **Portfolio Management**: Real-time balance and inventory tracking
- **Risk Management**: Position sizing and stop-loss mechanisms

### 🔬 **Analysis & Simulation**
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Overbought/Oversold
- **Monte Carlo Simulations**: Simple, Dynamic Volatility, Drift, Multivariate
- **Portfolio Optimization**: Modern Portfolio Theory implementation
- **Sentiment Analysis**: Bitcoin sentiment integration

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/QuantitativeTradingAI.git
cd QuantitativeTradingAI

# Install dependencies
pip install -r requirements.txt

# Setup environment
python main.py --setup
```

### Basic Usage

```bash
# Show project information
python main.py --info

# Start web application
python main.py --web

# Run model training
python main.py --train

# Run backtesting
python main.py --backtest
```

### Web Application

Start the Flask web server:
```bash
python main.py --web
```

Access the application at: `http://localhost:5000`

**Available Endpoints:**
- `/` - Main dashboard
- `/inventory` - Current holdings
- `/balance` - Account balance
- `/trade` - Execute trades
- `/reset` - Reset portfolio

## 📁 Project Structure

```
QuantitativeTradingAI/
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── Stock-Prediction-Models/        # Core models and agents
│   ├── agent/                      # 23 Trading agents
│   │   ├── 1.turtle-agent.ipynb
│   │   ├── 5.q-learning-agent.ipynb
│   │   ├── 14.actor-critic-agent.ipynb
│   │   └── ...
│   ├── deep-learning/              # 18 Deep learning models
│   │   ├── 1.lstm.ipynb
│   │   ├── 4.gru.ipynb
│   │   ├── 16.attention-is-all-you-need.ipynb
│   │   └── ...
│   ├── dataset/                    # Stock datasets
│   │   ├── GOOG.csv
│   │   ├── TSLA.csv
│   │   ├── BTC-sentiment.csv
│   │   └── ...
│   ├── realtime-agent/             # Real-time trading API
│   │   ├── app.py                  # Flask application
│   │   ├── model.pkl               # Trained model
│   │   └── *.csv                   # Stock data
│   ├── simulation/                 # Monte Carlo simulations
│   ├── stacking/                   # Ensemble methods
│   ├── misc/                       # Analysis and studies
│   └── output/                     # Generated plots and results
└── stock-forecasting-js/           # JavaScript frontend
```

## 📊 Model Performance

### Deep Learning Models
| Model | Accuracy | Training Time |
|-------|----------|---------------|
| LSTM | 95.69% | 01:09 |
| LSTM Bidirectional | 93.80% | 01:40 |
| GRU | 94.63% | 02:10 |
| Attention Transformer | 94.25% | 01:41 |
| Dilated CNN | 95.86% | 00:14 |

### Trading Agents Performance
All agents are tested on historical data with realistic trading constraints:
- Single unit transactions
- Transaction costs
- Slippage simulation
- Risk management rules

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# API Keys (optional)
ALPHA_VANTAGE_API_KEY=your_key_here
YAHOO_FINANCE_API_KEY=your_key_here

# Trading Parameters
INITIAL_CAPITAL=10000
TRANSACTION_COST=0.001
RISK_FREE_RATE=0.02

# Model Parameters
WINDOW_SIZE=20
LAYER_SIZE=500
LEARNING_RATE=0.001
```

### Model Configuration
Each model can be configured through its respective Jupyter notebook or configuration file. Key parameters include:

- **Window Size**: Historical data window for predictions
- **Layer Size**: Neural network architecture
- **Learning Rate**: Training optimization
- **Population Size**: Evolution strategy parameters
- **Risk Tolerance**: Trading agent behavior

## 📈 Usage Examples

### Training a Model

```python
# Example: Training LSTM model
from Stock_Prediction_Models.deep_learning.lstm import LSTMModel

model = LSTMModel(
    window_size=20,
    layer_size=500,
    learning_rate=0.001
)

model.train(
    data=stock_data,
    epochs=100,
    batch_size=32
)
```

### Running a Trading Agent

```python
# Example: Q-Learning Agent
from Stock_Prediction_Models.agent.q_learning import QLearningAgent

agent = QLearningAgent(
    initial_capital=10000,
    transaction_cost=0.001
)

results = agent.backtest(
    data=stock_data,
    episodes=1000
)
```

### Real-time Trading

```python
import requests

# Get current balance
response = requests.get('http://localhost:5000/balance')
balance = response.json()

# Execute a trade
trade_data = {
    'action': 'buy',
    'symbol': 'GOOG',
    'quantity': 10
}
response = requests.post('http://localhost:5000/trade', json=trade_data)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/QuantitativeTradingAI.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended to provide financial advice. Trading stocks involves risk, and past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 🙏 Acknowledgments

- Original Stock-Prediction-Models repository by [huseinzol05](https://github.com/huseinzol05/Stock-Prediction-Models)
- TensorFlow and PyTorch communities
- Financial data providers (Yahoo Finance, Alpha Vantage)
- Open-source contributors

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/QuantitativeTradingAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/QuantitativeTradingAI/discussions)
- **Email**: your.email@example.com

---

**Made with ❤️ for the quantitative trading community**
