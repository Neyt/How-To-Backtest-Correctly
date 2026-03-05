<div align="center">

# How To Backtest Correctly

### Stop Losing Money to Overfitted Backtests.

**The open-source implementation of Marcos Lopez de Prado's *Advances in Financial Machine Learning* methodologies.**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Neyt/How-To-Backtest-Correctly/issues)
[![Stars](https://img.shields.io/github/stars/Neyt/How-To-Backtest-Correctly?style=social)](https://github.com/Neyt/How-To-Backtest-Correctly/stargazers)

---

**Over 90% of backtested strategies fail in live trading.** This framework gives you the mathematical tools to know *before* you deploy.

[Get Started](#-quick-start) | [Documentation](#-core-modules) | [Contributing](#-contributing)

</div>

---

## Why This Exists

Most quant traders and asset managers commit the same fatal mistakes:

- They **overfit** strategies to historical noise and mistake it for signal
- They use **broken cross-validation** that leaks future information into training data
- They evaluate performance with a **naive Sharpe Ratio** that ignores multiple testing bias
- They label data using **fixed-time horizons** that ignore realistic market microstructure

**This repository** implements the complete scientific pipeline from Lopez de Prado's research to eliminate these pitfalls and build strategies that actually survive in production.

> *"Backtesting while researching is like drinking and driving. Do not research under the influence of a backtest."* -- Marcos Lopez de Prado

---

## Key Features

| Feature | What It Solves | Status |
|---|---|---|
| **Triple-Barrier Method** | Replaces naive fixed-horizon labels with path-aware, realistic labeling | Ready |
| **Meta-Labeling (Corrective AI)** | Splits side prediction from size/confidence -- dramatically improves F1-score | Ready |
| **Purging & Embargoing** | Eliminates information leakage in time-series cross-validation | Ready |
| **Combinatorial Purged CV (CPCV)** | Generates thousands of backtest paths instead of one fragile walk-forward test | Ready |
| **Deflated Sharpe Ratio (DSR)** | Corrects for multiple testing bias -- the #1 cause of false discoveries | Ready |
| **Probability of Backtest Overfitting (PBO)** | Quantifies the exact probability your strategy is curve-fitted | Ready |
| **Minimum Track Record Length** | Tells you how much data you *actually* need before trusting a result | Ready |

---

## Core Modules

### 1. The Two Laws of Quantitative Research

Before writing a single line of code, internalize these principles:

- **The First Law:** Focus efforts on researching *theories*, not backtesting trading rules. Feature importance is a research tool; backtesting is only for validation.
- **The Second Law ("Drinking and Driving"):** Never run a backtest until your model is fully specified. Backtesting while researching guarantees overfitting.

### 2. Labeling & Sizing: Triple-Barrier and Meta-Labeling

Stop using naive labels like "did price go up after 10 days?" -- they ignore the price path and real stop-out conditions.

- **Triple-Barrier Method:** Labels observations based on the *first barrier touched*:
  - Upper barrier (profit-taking)
  - Lower barrier (stop-loss)
  - Vertical barrier (max holding period)

- **Meta-Labeling (Corrective AI):** A two-model architecture that separates *direction* from *conviction*:
  - **Primary Model:** Decides the side (Long/Short) based on economic theory or a base strategy
  - **Secondary Model (Meta-Model):** Predicts whether the primary model will be *correct*, outputting a bet size between 0 and 1. Low confidence = skip the trade, preserve capital.

### 3. Data Splitting: Eliminating Information Leakage

Financial data is NOT independent and identically distributed (IID). Standard K-Fold CV and Walk-Forward are fundamentally broken for time series.

- **Purging:** Remove any training observation whose label overlaps in time with test set labels. Prevents the model from peeking at future outcomes.
- **Embargoing:** Remove a buffer of training data immediately *after* the test set to account for feature memory (e.g., ARMA processes).

### 4. The Backtest Engine: Combinatorial Purged Cross-Validation (CPCV)

Walk-Forward testing is dangerous -- it tests only a *single* historical path, making it trivially easy to overfit to one sequence of events.

**CPCV solves this by:**
1. Dividing time-series data into N sequential, non-overlapping groups
2. Generating ALL possible combinations of train/test splits
3. Applying Purging and Embargoing to every split
4. Producing **thousands of backtest paths** instead of one

**Result:** Instead of a single fragile Sharpe Ratio, you get the *entire distribution* of strategy performance across simulated market regimes.

### 5. Statistical Inference: Deflating the Sharpe Ratio

The standard Sharpe Ratio is broken. It assumes IID Normal returns and ignores how many strategies you tried before finding "the one."

- **Probabilistic Sharpe Ratio (PSR):** Adjusts for non-normal returns (skewness, kurtosis) and short samples. Frames the result as a probability.
- **Deflated Sharpe Ratio (DSR):** The gold standard. Corrects for Selection Bias under Multiple Testing by adjusting for:
  - Number of trials attempted (N)
  - Variance across simulated Sharpe Ratios
  - Sample length (T)
  - Return distribution shape (skewness & kurtosis)
- **Minimum Track Record Length (MinTRL):** The minimum number of observations needed to statistically confirm a strategy's Sharpe Ratio is above zero.

### 6. Strategy Evaluation & Overfitting Detection

- **Probability of Backtest Overfitting (PBO):** Using CPCV results, calculates the probability that your "optimal" in-sample configuration will underperform out-of-sample. **If PBO > 0.05, reject the strategy.**
- **Transaction Cost Integration:** Every backtest must include realistic slippage, commissions, and market impact models.

---

## Quick Start

```bash
git clone https://github.com/Neyt/How-To-Backtest-Correctly.git
cd How-To-Backtest-Correctly
pip install -r requirements.txt
```

```python
from afml import TripleBarrier, MetaLabeling, CPCV, DeflatedSharpe

# 1. Label your data properly
labels = TripleBarrier(close, events, pt_sl=[1, 2], min_ret=0.01)

# 2. Apply meta-labeling for bet sizing
meta_labels = MetaLabeling(primary_model, features, labels)

# 3. Validate with CPCV (not walk-forward!)
cpcv = CPCV(n_groups=10, k_test=2, purge_pct=0.01, embargo_pct=0.01)
results = cpcv.run(model, features, meta_labels)

# 4. Check if your strategy is real
dsr = DeflatedSharpe(results.sharpe_ratios, n_trials=100)
print(f"Deflated Sharpe Ratio: {dsr.value:.3f}")
print(f"Probability of Backtest Overfitting: {results.pbo:.2%}")
```

---

## Who Is This For?

- **Quantitative Traders** tired of strategies that work in backtests but die in production
- **Asset Managers** who need institutional-grade validation before deploying capital
- **Data Scientists** entering finance who want to avoid the most common statistical traps
- **Academics & Researchers** implementing AFML methodologies for reproducible research
- **Trading Educators** who want to teach their students the *right* way to build strategies

---

## Roadmap

- [x] Triple-Barrier Method
- [x] Meta-Labeling
- [x] Purging & Embargoing
- [x] Combinatorial Purged Cross-Validation
- [x] Deflated Sharpe Ratio & PSR
- [x] Probability of Backtest Overfitting
- [ ] Structural Breaks Detection (CUSUM filters)
- [ ] Fractionally Differentiated Features
- [ ] Entropy-based features (Shannon, Lempel-Ziv, plug-in)
- [ ] Market Microstructure features (VPIN, Kyle's Lambda)
- [ ] Portfolio Construction (HRP - Hierarchical Risk Parity)
- [ ] Full QuantConnect integration

---

## Acknowledgments & Credits

This repository is entirely based on the groundbreaking work of **[Prof. Marcos Lopez de Prado](https://www.quantresearch.org/)**, one of the most influential figures in modern quantitative finance.

All concepts, methodologies, and frameworks presented here originate from his research, including:

- **[Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)** (Wiley, 2018) -- The foundational text that introduced Triple-Barrier labeling, Meta-Labeling, CPCV, Deflated Sharpe Ratio, and the systematic approach to avoiding backtest overfitting.
- **[Machine Learning for Asset Managers](https://www.cambridge.org/core/elements/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F6F1D0F0C2D)** (Cambridge University Press, 2020) -- Further developments on portfolio construction and feature importance.
- His extensive body of **published research papers** on Combinatorial Purged Cross-Validation, Probability of Backtest Overfitting, Selection Bias under Multiple Testing, and meta-labeling.

**Prof. Lopez de Prado** is the Global Head of Quantitative Research & Development at Abu Dhabi Investment Authority (ADIA), a Professor of Practice at Cornell University, and has published over 50 scientific articles on quantitative finance and machine learning. His work has fundamentally changed how serious practitioners approach strategy development and validation.

> We strongly encourage anyone using this repository to **purchase and read his books**. This repo is a companion guide -- not a replacement for the depth and rigor of the original works.

**Official resources:**
- Website: [quantresearch.org](https://www.quantresearch.org/)
- Google Scholar: [Marcos Lopez de Prado](https://scholar.google.com/citations?user=5hG4UOEAAAAJ)
- Books: [Amazon Author Page](https://www.amazon.com/stores/Marcos-Lopez-de-Prado/author/B00O5XVIOQ)

---

## Contributing

Contributions are what make the open-source community thrive. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Star History

If this saved you from deploying an overfitted strategy, consider giving it a star. It helps others find it.

[![Star History Chart](https://api.star-history.com/svg?repos=Neyt/How-To-Backtest-Correctly&type=Date)](https://star-history.com/#Neyt/How-To-Backtest-Correctly&Date)

---

<div align="center">

**Built for traders who refuse to gamble with their capital.**

Made with discipline by [Ney Torres](https://github.com/Neyt)

</div>
