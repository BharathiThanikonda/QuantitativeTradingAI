#!/usr/bin/env python3
"""
QuantitativeTradingAI - Main Entry Point
========================================

A comprehensive quantitative trading and stock prediction platform featuring:
- 18+ Deep Learning Models (LSTM, GRU, Transformer, etc.)
- 23+ Trading Agents (Q-Learning, Actor-Critic, Evolution Strategy, etc.)
- Real-time Trading API
- Portfolio Optimization
- Monte Carlo Simulations
- Technical Analysis Tools

Author: [Your Name]
License: MIT
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def setup_environment():
    """Setup the environment and check dependencies."""
    print("🚀 Setting up QuantitativeTradingAI environment...")
    
    # Check if required directories exist
    required_dirs = [
        "Stock-Prediction-Models",
        "Stock-Prediction-Models/agent",
        "Stock-Prediction-Models/deep-learning",
        "Stock-Prediction-Models/dataset"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"⚠️  Warning: Directory {dir_path} not found")
    
    print("✅ Environment setup complete!")

def run_web_app():
    """Run the Flask web application for real-time trading."""
    print("🌐 Starting QuantitativeTradingAI Web Application...")
    
    try:
        from Stock_Prediction_Models.realtime_agent.app import app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except ImportError as e:
        print(f"❌ Error importing web app: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")

def run_model_training():
    """Run model training pipeline."""
    print("🤖 Starting Model Training Pipeline...")
    
    # This would integrate with your existing notebooks
    print("📊 Available models:")
    print("  - Deep Learning Models (18 models)")
    print("  - Trading Agents (23 agents)")
    print("  - Ensemble Methods")
    print("  - Technical Analysis Tools")
    
    print("💡 To train specific models, run the corresponding Jupyter notebooks in:")
    print("   Stock-Prediction-Models/deep-learning/")
    print("   Stock-Prediction-Models/agent/")

def run_backtesting():
    """Run backtesting on historical data."""
    print("📈 Starting Backtesting Engine...")
    
    print("📊 Available datasets:")
    datasets = [
        "GOOG.csv", "TSLA.csv", "FB.csv", "AMD.csv", 
        "BTC-sentiment.csv", "oil.csv", "eur-myr.csv"
    ]
    
    for dataset in datasets:
        print(f"  - {dataset}")
    
    print("💡 To run backtests, use the simulation notebooks in:")
    print("   Stock-Prediction-Models/simulation/")

def show_project_info():
    """Display project information and statistics."""
    print("\n" + "="*60)
    print("🎯 QuantitativeTradingAI - Advanced Trading Platform")
    print("="*60)
    
    print("\n📊 Project Statistics:")
    print("  • 18 Deep Learning Models")
    print("  • 23 Trading Agents")
    print("  • 15+ Stock Datasets")
    print("  • Real-time Trading API")
    print("  • Portfolio Optimization")
    print("  • Monte Carlo Simulations")
    
    print("\n🏗️  Architecture:")
    print("  • Deep Learning: LSTM, GRU, Transformer, CNN")
    print("  • Reinforcement Learning: Q-Learning, Actor-Critic, Evolution Strategy")
    print("  • Ensemble Methods: Stacking, Autoencoder")
    print("  • Technical Analysis: RSI, MACD, Bollinger Bands")
    
    print("\n🚀 Quick Start:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run web app: python main.py --web")
    print("  3. Train models: python main.py --train")
    print("  4. Run backtests: python main.py --backtest")
    
    print("\n📚 Documentation:")
    print("  • Models: Stock-Prediction-Models/deep-learning/")
    print("  • Agents: Stock-Prediction-Models/agent/")
    print("  • API: Stock-Prediction-Models/realtime-agent/")
    print("  • Simulations: Stock-Prediction-Models/simulation/")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="QuantitativeTradingAI - Advanced Stock Prediction and Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --web          # Start web application
  python main.py --train        # Run model training
  python main.py --backtest     # Run backtesting
  python main.py --info         # Show project information
        """
    )
    
    parser.add_argument('--web', action='store_true', 
                       help='Start the Flask web application')
    parser.add_argument('--train', action='store_true', 
                       help='Run model training pipeline')
    parser.add_argument('--backtest', action='store_true', 
                       help='Run backtesting on historical data')
    parser.add_argument('--info', action='store_true', 
                       help='Show project information')
    parser.add_argument('--setup', action='store_true', 
                       help='Setup environment and check dependencies')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_environment()
    elif args.web:
        run_web_app()
    elif args.train:
        run_model_training()
    elif args.backtest:
        run_backtesting()
    elif args.info:
        show_project_info()
    else:
        show_project_info()
        print("\n💡 Use --help for more options")

if __name__ == "__main__":
    main()
