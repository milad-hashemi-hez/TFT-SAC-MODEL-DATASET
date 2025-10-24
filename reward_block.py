
# reward_block.py

import numpy as np

def default_reward_block(log_return, position, action, last_action, frame, 
                        net_worth, max_equity, returns_history):
    """
    Default reward block - exactly the same as your current reward logic
    """
    reward = log_return * 1.0

    # --- Bonus/Penalty Rewards ---
    bonus_reward = 0.0
    # Reward for holding during strong trend
    if position > 0 and frame['trend_confidence'] > 0.4:
        bonus_reward += 0.05
    # Penalize aggressive trading during high volatility
    if abs(action) > 0.3 and frame['high_volatility_regime'] > 0.5:
        bonus_reward -= 0.05
    # Reward for holding during breakout
    if position > 0 and frame['breakout_strength'] > 0.3:
        bonus_reward += 0.03
    # Reward for stable actions (low churn)
    if position > 0 and abs(action - last_action) < 0.1:
        bonus_reward += 0.01

    reward += bonus_reward

    # --- Drawdown Penalty ---
    drawdown = (net_worth - max_equity) / max_equity if max_equity > 0 else 0.0
    if drawdown < -0.20:
        reward -= 0.5
    elif drawdown < -0.10:
        reward -= 0.2

    # --- Sharpe Ratio Reward Shaping ---
    if len(returns_history) >= 50:
        recent_returns = np.array(returns_history[-50:])
        sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)
        reward += np.clip(sharpe, -0.05, 0.1) * 0.1

    return float(reward)