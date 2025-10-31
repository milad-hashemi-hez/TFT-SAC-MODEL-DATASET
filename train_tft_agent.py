
# train_tft_agent.py

import numpy as np
import matplotlib.pyplot as plt
import torch
from trading_environment import BitcoinTradingEnv
from TFTSACAgent import TFTSACAgent

# Initialize environment and agent
env = BitcoinTradingEnv("bitcoin_train.csv")

print("🔍 DATA INTEGRITY CHECK:")
print(f"Training file: bitcoin_train.csv")
print(f"Data shape: {env.df.shape}")
print(f"Date range: {env.df['Open time'].min()} to {env.df['Open time'].max()}")
print(f"Total days: {len(env.df)}")
print(f"First 5 dates: {env.df['Open time'].head(5).tolist()}")
print(f"Last 5 dates: {env.df['Open time'].tail(5).tolist()}")

state_size = env.state_size  # ✅ Automatically detected
action_size = 1

print(f"🎯 STATE SIZE: {state_size} (automatically detected from environment)")

# TFT-SAC Hyperparameters
agent = TFTSACAgent(
    state_size=state_size,
    action_size=action_size,
    actor_lr=3e-5,
    critic_lr=4.8e-5,
    gamma=0.99,
    alpha=0.2,
    tau=0.005,
    batch_size=128,
    seq_len=30,
    hidden_size=128,
    num_heads=8
)

# Training config
episodes = 500
collect_steps_per_update = 512
reward_history = []
net_worth_history = []
roi_history = []
update_count = 0

# Add verification step
print("🧪 VERIFICATION:")
state = env.reset()
print(f"State shape: {state.shape}")
print(f"State range: [{state.min():.3f}, {state.max():.3f}]")

# Loss tracking
forecast_losses = []
actor_losses = []
critic_losses = []
alphas = []

# Track best model
best_net_worth = env.initial_balance
best_episode = 0

# Live plotting - FIGURE 1: Reward & Net Worth (Lines)
plt.ion()
fig1, ax1 = plt.subplots(figsize=(12, 6))
line_reward, = ax1.plot([], [], label="Total Reward", color='blue', alpha=0.7)
line_net_worth, = ax1.plot([], [], label="Net Worth ($)", color='green', linewidth=2)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Value")
ax1.set_title("Live TFT-SAC Training — Reward & Net Worth")
ax1.grid(True, alpha=0.3)
ax1.legend()

# FIGURE 2: Net Worth Distribution (Dots)
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.set_title('Episode Net Worth Distribution (Dots)')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Net Worth ($)')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, episodes)
ax2.set_ylim(0, 50000)  # Adjust if your net worth goes higher

episode_steps = 0

# Main training loop
for e in range(episodes):
    state = env.reset()
    
    total_reward = 0
    done = False
    step_count = 0

    state_buffer = []
    next_state_buffer = []

    while not done and step_count < collect_steps_per_update:
        state_buffer.append(state)
        if len(state_buffer) > agent.seq_len:
            state_buffer.pop(0)

        action, log_prob = agent.act(np.array(state_buffer), evaluate=False)

        next_state, reward, done, _ = env.step(action)

        next_state_buffer.append(next_state)
        if len(next_state_buffer) > agent.seq_len:
            next_state_buffer.pop(0)

        if len(state_buffer) == agent.seq_len and len(next_state_buffer) == agent.seq_len:
            agent.remember(
                state_seq=np.array(state_buffer),
                action=action,
                reward=reward,
                next_state_seq=np.array(next_state_buffer),
                done=done
            )

        state = next_state
        total_reward += reward
        step_count += 1
        episode_steps += 1

        if env.net_worth <= 100:
            done = True
            print("🚨 BANKRUPT — Stopping episode early.")

        if done:
            break

    # Log metrics
    roi = (env.net_worth - env.initial_balance) / env.initial_balance * 100
    reward_history.append(total_reward)
    net_worth_history.append(env.net_worth)
    roi_history.append(roi)

    print(f"Episode {e+1}/{episodes} - ROI: {roi:+.2f}% | Net Worth: ${env.net_worth:,.2f} | Reward: {total_reward:.1f} | Steps: {step_count}")

    # Update Figure 1: Line plot
    x_data = list(range(1, len(reward_history) + 1))
    line_reward.set_xdata(x_data)
    line_reward.set_ydata(reward_history)
    line_net_worth.set_xdata(x_data)
    line_net_worth.set_ydata(net_worth_history)
    ax1.relim()
    ax1.autoscale_view()
    fig1.canvas.draw()
    fig1.canvas.flush_events()

    # Update Figure 2: Scatter plot (dots)
    ax2.clear()
    ax2.set_title('Episode Net Worth Distribution (Dots)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Net Worth ($)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, episodes)
    ax2.set_ylim(0, 50000)

    colors = ['green' if nw > env.initial_balance else 'red' for nw in net_worth_history]
    ax2.scatter(x_data, net_worth_history, c=colors, alpha=0.7, s=30, edgecolors='black', linewidth=0.5)

    fig2.canvas.draw()
    fig2.canvas.flush_events()

    # Train agent
    if episode_steps >= collect_steps_per_update or (done and len(agent.memory) >= agent.batch_size):
        result = agent.train()
        if result is not None:
            actor_loss, critic_loss, alpha, forecast_loss = result
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            alphas.append(alpha)
            forecast_losses.append(forecast_loss)
            
            print(f"--- TFT-SAC Updated (#{update_count + 1}) ---")
            print(f"  → Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Alpha: {alpha:.4f}, Forecast Loss: {forecast_loss:.4f}")
            update_count += 1
        episode_steps = 0

    # Save best model
    if env.net_worth > best_net_worth:
        best_net_worth = env.net_worth
        best_episode = e + 1
        agent.save_model("best_tft_sac_model.pth")
        print(f"🏆 NEW BEST MODEL SAVED! Net Worth: ${env.net_worth:,.2f} (Episode {e+1})")

    # Save checkpoint every 100 episodes
    if (e + 1) % 100 == 0:
        agent.save_model(f"tft_sac_model_ep{e+1}.pth")
        print(f"💾 Checkpoint saved at episode {e+1}")

# ===== PROFIT & NET WORTH STATISTICS (Refined) =====
net_worths = np.array(net_worth_history)
profits = net_worths - env.initial_balance

# Metrics
avg_net_worth = np.mean(net_worths)
avg_profit = np.mean(profits)
std_net_worth = np.std(net_worths)
std_profit = np.std(profits)

# Win/Loss count
negative_episodes = np.sum(profits < 0)
total_episodes = len(profits)
negative_percent = (negative_episodes / total_episodes) * 100
profitable_percent = 100.0 - negative_percent

print("\n📊 EPISODE PERFORMANCE STATISTICS:")
print(f"Average Net Worth per Episode: ${avg_net_worth:,.2f}")
print(f"Average Profit per Episode:    ${avg_profit:,.2f}")
print(f"Standard Deviation (Net Worth): ${std_net_worth:,.2f}")
print(f"Standard Deviation (Profit):    ${std_profit:,.2f}")
print(f"Your agent lost money in only {negative_episodes} of {total_episodes} episodes ({negative_percent:.1f}%).")
print(f"{total_episodes - negative_episodes} of {total_episodes} episodes were profitable ({profitable_percent:.1f}%).")
# ===================================================

# Finalize plots
plt.ioff()
plt.show()

# Summary
print("\n✅ TFT-SAC TRAINING COMPLETE!")
print(f"Final Net Worth: ${env.net_worth:,.2f}")
print(f"Best Net Worth: ${best_net_worth:,.2f} (Episode {best_episode})")
print(f"Final ROI: {roi_history[-1]:+.2f}%")
print(f"Average Reward (Last 10): {np.mean(reward_history[-10:]):.2f}")
if alphas:
    print(f"Max Alpha (Entropy): {max(alphas):.4f}")

# Forecast performance
if forecast_losses:
    print(f"Final Forecast Loss: {forecast_losses[-1]:.4f}")
    print(f"Average Forecast Loss: {np.mean(forecast_losses):.4f}")

# Save final model
agent.save_model("final_tft_sac_model.pth")
print("\n💾 Final model saved: final_tft_sac_model.pth")

# Plot losses
if actor_losses:
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 4, 1)
    plt.plot(actor_losses, color='red')
    plt.title("Actor Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 2)
    plt.plot(critic_losses, color='orange')
    plt.title("Critic Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 3)
    plt.plot(alphas, color='purple')
    plt.title("Entropy Tuning (Alpha)")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 4)
    plt.plot(forecast_losses, color='green')
    plt.title("Forecast Loss")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()