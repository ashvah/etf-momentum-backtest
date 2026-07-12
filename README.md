# ETF Momentum Rotation Backtest

A reproducible Python backtest of a monthly momentum rotation strategy across U.S. ETFs.


> **Status:** Work in progress

## Overview

This project studies a simple cross-asset momentum strategy:

## Default Asset Universe

The project provides the following default ETF universe:

| Ticker | Exposure                              |
| ------ | ------------------------------------- |
| SPY    | U.S. large-cap equities               |
| QQQ    | U.S. growth and technology equities   |
| TLT    | Long-term U.S. Treasury bonds         |
| IEF    | Intermediate-term U.S. Treasury bonds |
| GLD    | Gold                                  |

The universe is configurable. Users may provide a different set of ticker symbols through the Python configuration object or the command-line interface.

## Project Structure

```text
etf-momentum-backtest/
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── notebooks/
├── outputs/
│   ├── figures/
│   │   └── .gitkeep
│   └── results/
│       └── .gitkeep
├── src/
│   └── etf_momentum_backtest/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config.py
│       └── cli.py
├── tests/
│   ├── test_config.py
│   └── test_cli.py
├── .gitignore
├── pyproject.toml
├── README.md
└── LICENSE
```

## Requirements

* `pip`;
* Git is recommended for version control.

Core dependencies include:

* NumPy;
* pandas;
* Matplotlib;
* PyArrow;
* yfinance.

Development and research tools include:

* pytest;
* Ruff;
* JupyterLab.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/ashvah/etf-momentum-backtest.git
cd etf-momentum-backtest
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project with development and research dependencies:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev,research]"
```

The editable installation means that changes under `src/` take effect without reinstalling the package.

## Usage

### Display the default configuration

```bash
etf-momentum
```

Equivalent module command:

```bash
python -m etf_momentum_backtest
```

At the current stage, the command validates and displays the experiment configuration. It does not yet execute the full backtest.

### Customize the experiment

```bash
etf-momentum \
    --tickers SPY QQQ GLD TLT \
    --start-date 2010-01-01 \
    --end-date 2025-12-31 \
    --lookback-days 252 \
    --top-k 2 \
    --cost-bps 5 \
    --initial-capital 500000
```

Available command-line parameters include:

| Argument            | Meaning                    |             Default |
| ------------------- | -------------------------- | ------------------: |
| `--tickers`         | ETF ticker symbols         | SPY QQQ TLT IEF GLD |
| `--start-date`      | Backtest start date        |          2005-01-01 |
| `--end-date`        | Optional end date          |    latest available |
| `--lookback-days`   | Momentum lookback period   |                 126 |
| `--top-k`           | Number of selected assets  |                   2 |
| `--cost-bps`        | One-way transaction cost   |                  10 |
| `--initial-capital` | Starting portfolio capital |           1,000,000 |

## Development

Run the test suite:

```bash
pytest
```

Check the code:

```bash
ruff check .
```

Automatically fix supported linting issues:

```bash
ruff check . --fix
```

## Disclaimer

This project is for educational and research purposes only.

It does not constitute investment advice and is not intended for live trading without substantially more data validation, execution modeling, risk controls, and independent review.
