
# reward_block.py

import numpy as np

def default_reward_block(log_return, position, action, last_action, frame, 
                        net_worth, max_equity, returns_history):
    """
    Default reward block - exactly the same as your current reward logic
    """
    reward = log_return * 100.0

    return float(reward)