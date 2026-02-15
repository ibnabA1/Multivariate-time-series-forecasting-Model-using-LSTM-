# Multivariate Time Series Forecasting using LSTM

This project builds a deep learning pipeline to analyze historical stock data and forecast future behavior using a Long Short-Term Memory (LSTM) neural network in PyTorch.  
Instead of treating each day independently, the model studies how values evolve across time and learns patterns from the past 90 trading days to make predictions.

---

## Why This Approach

Financial and operational data are sequential by nature.  
Today's value depends on what happened yesterday, last week, and even months ago.

Traditional machine learning models treat rows as separate observations.  
An LSTM is designed specifically for sequences — it remembers previous context and uses it to understand trends, momentum, and recurring patterns.

This project demonstrates how sequence modeling can capture:

- Trend continuation
- Sudden spikes in activity
- Cyclical behavior
- Long-term dependencies

---

## Data Processing Logic

1. Historical data is downloaded using `yfinance`
2. A moving average indicator is added to provide trend information
3. Missing values from rolling calculations are handled
4. Features are normalized using MinMax scaling
5. A sliding window of 90 days is created

The sliding window is important:  
Each training example contains 90 consecutive days, allowing the model to understand how patterns develop over time instead of learning isolated points.

---

## Model Design

The model is a 2-layer LSTM network followed by a fully connected layer.

**How it works conceptually:**

The network reads daily values one step at a time.  
Each day updates its memory of what is happening in the sequence.  
After the last day, the model has a summarized understanding of recent behavior and produces the prediction.

So instead of predicting from a single row:

> It predicts based on recent history.

---

## Training Strategy

- Dataset split into train / validation / test
- Mean Squared Error used as loss
- Adam optimizer for stable learning
- Best model saved based on validation loss
- Training curves plotted to monitor convergence

This prevents overfitting and ensures the model generalizes to unseen data.

---

## Predictions

The project performs three types of predictions:

1. Reconstructing known data to verify learning quality
2. Predicting unseen test sequences
3. Forecasting the next 90 future days using recursive prediction

Future forecasting works by feeding each new prediction back into the model as the next input — similar to how real-world forecasting systems operate.

---

## Business Value

This project is not limited to stocks.  
The same pipeline can be applied anywhere historical behavior influences the future.

Examples:

**Finance**
- Stock movement forecasting
- Risk trend monitoring
- Trading signal analysis

**Telecommunications**
- Network traffic prediction
- Capacity planning

**Retail & E-commerce**
- Demand forecasting
- Inventory optimization

**Manufacturing**
- Equipment usage prediction
- Predictive maintenance scheduling

**Energy**
- Electricity consumption forecasting
- Load balancing

Any time-dependent numeric data can use this approach.

---

## Tech Stack

- Python
- PyTorch
- Pandas & NumPy
- Scikit-learn
- Matplotlib
- yfinance

---

## Key Takeaway

The goal of this project is not just predicting prices, but demonstrating how deep learning models can learn behavior from historical sequences and make informed future estimates — a core requirement in real-world decision systems.
