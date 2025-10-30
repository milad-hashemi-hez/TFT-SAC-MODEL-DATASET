
# validate_agent.py

import numpy as np
import pandas as pd
from trading_environment import BitcoinTradingEnv
from TFTSACAgent import TFTSACAgent



def validate_trained_model(model_path="best_tft_sac_model.pth", test_data_path="bitcoin_test.csv"):
    """
    Proper validation of trained TFT-SAC agent on test data
    """
    
    print("🧪 STARTING VALIDATION...")
    
    # Initialize test environment
    test_env = BitcoinTradingEnv(test_data_path, normalize_obs=False)  # Turn off normalization for consistent evaluation
    state_size = test_env.state_size
    action_size = 1
    
    # Initialize agent with same architecture
    agent = TFTSACAgent(
        state_size=state_size,
        action_size=action_size,
        seq_len=30,  # Same as training
        hidden_size=128,
        num_heads=8
    )
    
    # Load the trained model
    print(f"📥 Loading model: {model_path}")
    agent.load_model(model_path)
    
    # Validation run
    print("🔄 Running validation on test data...")
    
    state = test_env.reset()
    total_reward = 0
    done = False
    step_count = 0
    
    # Store trading history for analysis
    trading_history = []
    state_buffer = []
    
    while not done:
        # Build state sequence (same as training)
        state_buffer.append(state)
        if len(state_buffer) > agent.seq_len:
            state_buffer.pop(0)
        
        if len(state_buffer) == agent.seq_len:
            # Get action from trained agent (no exploration)
            action, _ = agent.act(np.array(state_buffer), evaluate=True)
            
            # Execute action
            next_state, reward, done, info = test_env.step(action)
            
            # Record trade info
            trading_history.append({
                'step': step_count,
                'net_worth': test_env.net_worth,
                'action': action,
                'reward': reward,
                'position': test_env.position,
                'balance': test_env.balance
            })
            
            total_reward += reward
            state = next_state
            step_count += 1
            
            # Progress indicator
            if step_count % 10 == 0:
                print(f"Step {step_count}: Net Worth = ${test_env.net_worth:,.2f}")
        else:
            # Not enough history yet, take neutral action
            next_state, reward, done, info = test_env.step(0.0)
            state = next_state
            step_count += 1
    
    # Calculate performance metrics
    initial_balance = test_env.initial_balance
    final_net_worth = test_env.net_worth
    total_return = (final_net_worth - initial_balance) / initial_balance * 100
    
    print("\n" + "="*50)
    print("📊 VALIDATION RESULTS")
    print("="*50)
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Net Worth: ${final_net_worth:,.2f}")
    print(f"Total Return: {total_return:+.2f}%")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Steps Completed: {step_count}")
    print(f"Total Trades: {len(test_env.trades)}")
    print(f"Final Position: {test_env.position:.6f} BTC")
    print(f"Final Cash: ${test_env.balance:,.2f}")
    
    # Analyze trades
    if test_env.trades:
        print(f"\n📈 TRADING ANALYSIS:")
        buys = [t for t in test_env.trades if t[0] == 'buy']
        sells = [t for t in test_env.trades if t[0] == 'sell']
        print(f"Buy trades: {len(buys)}")
        print(f"Sell trades: {len(sells)}")
        
        if buys:
            avg_buy_price = sum(t[3] for t in buys) / len(buys)
            print(f"Average buy price: ${avg_buy_price:.2f}")
        
        if sells:
            avg_sell_price = sum(t[3] for t in sells) / len(sells)
            print(f"Average sell price: ${avg_sell_price:.2f}")
    
    # Compare with buy-and-hold strategy
    print(f"\n📈 BENCHMARK COMPARISON:")
    buy_hold_return = (test_env.df['Close'].iloc[-1] / test_env.df['Close'].iloc[0] - 1) * 100
    print(f"Buy & Hold Return: {buy_hold_return:+.2f}%")
    print(f"Agent vs Buy&Hold: {total_return - buy_hold_return:+.2f}%")
    
    return {
        'final_net_worth': final_net_worth,
        'total_return': total_return,
        'total_reward': total_reward,
        'trades': test_env.trades,
        'trading_history': trading_history,
        'buy_hold_return': buy_hold_return
    }

def validate_multiple_models():
    """Validate multiple saved models to find the best one"""
    
    models_to_test = [
        "best_tft_sac_model.pth",
        "final_tft_sac_model.pth", 
        "tft_sac_model_ep200.pth",
        "tft_sac_model_ep300.pth",
        "tft_sac_model_ep400.pth",
        "tft_sac_model_ep500.pth"
    ]
    
    results = {}
    
    for model_file in models_to_test:
        try:
            print(f"\n🔍 Testing {model_file}...")
            result = validate_trained_model(model_file)
            results[model_file] = result
        except Exception as e:
            print(f"❌ Failed to test {model_file}: {e}")
    
    # Find best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['final_net_worth'])
        print(f"\n🏆 BEST MODEL: {best_model[0]}")
        print(f"   Net Worth: ${best_model[1]['final_net_worth']:,.2f}")
        print(f"   Return: {best_model[1]['total_return']:+.2f}%")

if __name__ == "__main__":
    # Validate your best model
    print("🚀 TFT-SAC AGENT VALIDATION")
    print("This will test your trained model on UNSEEN test data")
    print("="*60)
    
    # Test single model
    results = validate_trained_model("best_tft_sac_model.pth")
    
    # Uncomment to test multiple models
    # validate_multiple_models()