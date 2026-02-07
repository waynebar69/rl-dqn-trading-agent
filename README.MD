# Reinforcement Learning Trading Agent (DQN)

## Overview

This project implements an end-to-end Deep Q-Network (DQN) trading agent trained on historical equity price data. The focus is on building and wiring together the full RL pipeline (from data preparation and environment design to agent training, evaluation, and testing) rather than on producing a profitable strategy.

The implementation closely follows a fixed project specification and marking rubric. As a result, the emphasis is on correct RL mechanics, clean data handling, and reproducible training workflows, mirroring the constraints often encountered in real-world research and production environments.

## Project Objectives
- Design a simulated trading environment using historical market data
- Implement a DQN agent from scratch in Keras
- Encode action selection using ε-greedy exploration
- Train the agent using experience replay and the Bellman equation
- Evaluate agent behaviour on a held-out test dataset
- Demonstrate understanding of how RL components interact as a system

**Note:** This project is not intended to produce a profitable trading strategy. Instead, it serves as a foundation for future work involving improved reward design, longer training horizons, richer state representations, and more realistic backtesting assumptions.

## Data Preparation

### Market Data
- Historical OHLC price data
- Missing values forward-filled
- Visual inspection of raw vs filled data

### Feature Engineering
- Filtered to Close prices only (as required)
- Computed 20-day Bollinger Bands:
    - Moving Average
    - Upper Band
    - Lower Band

### Normalization
- Each feature normalized using **StandardScaler**
- Ensures stable training behaviour for the neural network

### Train / Test Split
- Dataset split 50 / 50
- First half used for training
- Second half held out for evaluation
- Explicit checks to confirm no overlap
- Data converted to NumPy arrays for Keras compatibility

### Environment & State Representation
The trading environment presents the agent with a **2-day rolling state**, consisting of:
- Today’s:
    - Close price
    - Bollinger Band features
- Yesterday’s:
    - Close price
    - Bollinger Band features

This state representation captures short-term price movement and volatility context while keeping the state space intentionally small.

## Agent Architecture

### Deep Q-Network (DQN)
- Fully connected neural network built from scratch using Keras
- Architecture and hyperparameters fixed per project specification:
    - Dense layers
    - ReLU activations
    - Mean Squared Error loss
    - Adam optimizer with fixed learning rate

### Action Space
Discrete actions:
- Buy
- Hold
- Sell

### Action Policy
- ε-greedy exploration strategy
- Balances exploration vs exploitation during training
- Greedy policy used during testing

### Experience Replay & Training Logic
- Fixed-size replay buffer
- Mini-batch sampling during training
- Target Q-values updated using the Bellman equation
- Model fitted on each mini-batch
- Exploration rate decayed over time

This mirrors the standard DQN training loop used in many applied RL systems.

### Training & Evaluation

#### Training Loop
- Iterates through historical price data
- At each timestep:
    - Construct current state
    - Select action via ε-greedy policy
    - Compute reward based on trading outcome
    - Store experience in replay buffer
    - Perform experience replay update

#### Testing Loop
- Loads trained model checkpoint
- Runs agent in test mode (no exploration)
- Evaluates behaviour on unseen data

Performance is assessed qualitatively via trade decisions and portfolio evolution rather than profitability.

## Key Takeaways
- Reinforcement learning for trading is primarily a systems engineering problem
- Small design choices in state representation, reward definition, and training loops have outsized effects
- Naive reward structures and short training horizons are insufficient for real-world deployment
- This project provides a solid base for more advanced extensions:
    - Alternative reward functions
    - Risk-aware constraints (drawdowns, position sizing)
    - Longer training horizons
    - More expressive state spaces
    - Offline evaluation and walk-forward testing

## Future Extensions (Out of Scope for This Project)
- Risk-adjusted reward functions
- Transaction costs and slippage
- Longer training runs on larger datasets
- Target networks / Double DQN
- Portfolio-level RL agents

## Disclaimer

This project is for educational and research purposes only and does not constitute financial advice or a deployable trading strategy.