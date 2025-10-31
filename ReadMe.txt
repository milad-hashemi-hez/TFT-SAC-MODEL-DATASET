
TFT-SAC Bitcoin Trading Agent

A sophisticated deep reinforcement learning system combining Temporal Fusion Transformers (TFT) with Soft Actor-Critic (SAC) for algorithmic trading on Bitcoin markets.

🎯 Core Architecture
Hybrid Model: TFT + SAC
Temporal Fusion Transformer (TFT): Processes multi-scale temporal patterns with attention mechanisms and forecasting head

Soft Actor-Critic (SAC): Maximum entropy RL for robust policy learning with automatic temperature tuning

Database :
Bitcoin dataset 2016 - 2025 ( related modules to download and clean and divide data)
divided to train and test part

requerments : 
include all requerments and packages

trading_environment :
State Space (50 Features)

training LOOP include :
normal graph
dots ( distribution graph )
Losses graph


📊 EPISODE PERFORMANCE STATISTICS:
Average Net Worth per Episode: $17,753.69
Average Profit per Episode:    $7,753.69
Standard Deviation (Net Worth): $4,429.93
Standard Deviation (Profit):    $4,429.93
Your agent lost money in only 2 of 500 episodes (0.4%).
498 of 500 episodes were profitable (99.6%).

✅ TFT-SAC TRAINING COMPLETE!
Final Net Worth: $17,244.35
Best Net Worth: $40,617.29 (Episode 176)
Final ROI: +72.44%
Average Reward (Last 10): 57.85
Max Alpha (Entropy): 1.0100
Final Forecast Loss: 0.0568
Average Forecast Loss: 0.1869


🚀 TFT-SAC AGENT VALIDATION
This will test your trained model on UNSEEN test data
============================================================
🧪 STARTING VALIDATION...
📊 Dataset shape after cleaning: (76, 69)
📥 Loading model: best_tft_sac_model.pth
🔄 Running validation on test data...
Step 30: Net Worth = $10,000.00
Step 40: Net Worth = $10,000.00
Step 50: Net Worth = $10,000.00
Step 60: Net Worth = $10,000.00
Step 70: Net Worth = $10,006.46

==================================================
📊 VALIDATION RESULTS
==================================================
Initial Balance: $10,000.00
Final Net Worth: $10,006.46
Total Return: +0.06%
Total Reward: 0.06
Steps Completed: 75
Total Trades: 46
Final Position: 0.000000 BTC
Final Cash: $10,006.46

📈 TRADING ANALYSIS:
Buy trades: 1
Sell trades: 45
Average buy price: $123510.45
Average sell price: $114555.50

📈 BENCHMARK COMPARISON:
Buy & Hold Return: -7.11%
Agent vs Buy&Hold: +7.18%

