
# train_tft_agent.py
import numpy as np
import matplotlib.pyplot as plt
import torch
from trading_environment import BitcoinTradingEnv
from TFTSACAgent import TFTSACAgent

# Initialize environment and agent
env = BitcoinTradingEnv("bitcoin_data.csv")
state_size = 24
action_size = 1

# TFT-SAC Hyperparameters
agent = TFTSACAgent(
    state_size=state_size,
    action_size=action_size,
    lr=3e-4,
    gamma=0.99,
    alpha=0.2,
    tau=0.005,
    batch_size=128,
    seq_len=20,
    hidden_size=128,
    num_heads=8  # Number of attention heads in TFT
)

# Training config
episodes = 500
collect_steps_per_update = 128
reward_history = []
net_worth_history = []
roi_history = []
update_count = 0

# Track best model
best_net_worth = env.initial_balance
best_episode = 0

# Live plotting
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
line_reward, = ax.plot([], [], label="Total Reward", color='blue', alpha=0.7)
line_net_worth, = ax.plot([], [], label="Net Worth ($)", color='green', linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel("Value")
ax.set_title("Live TFT-SAC Training — Reward & Net Worth")
ax.grid(True, alpha=0.3)
ax.legend()

# Loss tracking
actor_losses, critic_losses, alphas = [], [], []

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

        # Store experience only if we have full sequences
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

    # Update live plot
    x_data = list(range(1, len(reward_history) + 1))
    line_reward.set_xdata(x_data)
    line_reward.set_ydata(reward_history)
    line_net_worth.set_xdata(x_data)
    line_net_worth.set_ydata(net_worth_history)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)

    # Train agent if enough data collected
    if episode_steps >= collect_steps_per_update or (done and len(agent.memory) >= agent.batch_size):
        result = agent.train()
        if result is not None:
            actor_loss, critic_loss, alpha = result
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            alphas.append(alpha)
            print(f"--- TFT-SAC Updated (#{update_count + 1}) ---")
            if actor_loss is not None:
                print(f"  → Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Alpha: {alpha:.4f}")
            else:
                print("  → Not enough memory to train yet")
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

# Finalize plot
plt.ioff()
plt.show()

# Summary
print("\n✅ TFT-SAC TRAINING COMPLETE!")
print(f"Final Net Worth: ${env.net_worth:,.2f}")
print(f"Best Net Worth: ${best_net_worth:,.2f} (Episode {best_episode})")
print(f"Final ROI: {roi_history[-1]:+.2f}%")
print(f"Average Reward (Last 10): {np.mean(reward_history[-10:]):.2f}")
print(f"Max Alpha (Entropy): {max(alphas):.4f}")

# Save final model
agent.save_model("final_tft_sac_model.pth")
print("\n💾 Final model saved: final_tft_sac_model.pth")

# Plot losses
if len(actor_losses) > 0:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(actor_losses, label="Actor Loss", color='red')
    plt.title("Actor Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(critic_losses, label="Critic Loss", color='orange')
    plt.title("Critic Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(alphas, label="Alpha (Entropy Coeff)", color='purple')
    plt.title("Entropy Tuning (Alpha)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()