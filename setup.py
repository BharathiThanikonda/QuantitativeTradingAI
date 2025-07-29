#!/usr/bin/env python3
"""
Setup script for QuantitativeTradingAI
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantitative-trading-ai",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive quantitative trading and stock prediction platform",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/QuantitativeTradingAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0",
            "torch-cuda>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantitative-trading-ai=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.json", "*.yaml", "*.yml"],
    },
    keywords=[
        "trading",
        "stock-prediction",
        "machine-learning",
        "deep-learning",
        "reinforcement-learning",
        "quantitative-finance",
        "lstm",
        "gru",
        "transformer",
        "q-learning",
        "actor-critic",
        "evolution-strategy",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/QuantitativeTradingAI/issues",
        "Source": "https://github.com/yourusername/QuantitativeTradingAI",
        "Documentation": "https://github.com/yourusername/QuantitativeTradingAI#readme",
    },
) 