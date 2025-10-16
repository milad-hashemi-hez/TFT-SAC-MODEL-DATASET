# trading_environment.py

import numpy as np
import pandas as pd
import ta

# ==============================================================================
# RunningMeanStd: Online algorithm to compute mean and variance incrementally.
# Used for normalizing observations in reinforcement learning to stabilize training.
# Maintains running estimates of mean and variance across batches of data.
# ==============================================================================
class RunningMeanStd:
    def __init__(self, shape):
        # Initialize mean (zeros), variance (ones), and sample count (zero)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x):
        # Update running mean and variance with a new batch of data `x`
        # Uses Welford's online algorithm for numerical stability
        x = np.array(x, dtype=np.float64)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        # Normalize input `x` using current running mean and std (with epsilon for stability)
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def reset(self):
        # Reset statistics to initial state (used during environment reset)
        self.mean = np.zeros_like(self.mean)
        self.var = np.ones_like(self.var)
        self.count = 0


# ==============================================================================
# BitcoinTradingEnv: A custom OpenAI Gym-style trading environment for Bitcoin.
# Simulates trading with realistic mechanics: fees, position sizing, net worth,
# and reward shaping based on risk-adjusted returns and market conditions.
# ==============================================================================
class BitcoinTradingEnv:
    def __init__(self, df_or_path, initial_balance=10000, transaction_fee=0.001, normalize_obs=True, clip_rewards=True):
        # Load data: accept either a DataFrame or a CSV file path
        if isinstance(df_or_path, str):
            self.df = pd.read_csv(df_or_path)
            self.df['Open time'] = pd.to_datetime(self.df['Open time'])
        else:
            self.df = df_or_path

        # Trading parameters
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.normalize_obs = normalize_obs
        self.clip_rewards = clip_rewards  # Note: this flag is stored but not currently used in reward computation

        # --- Technical Indicators ---
        # RSI (Relative Strength Index): momentum oscillator (0–100), indicates overbought/oversold
        self.df["rsi"] = ta.momentum.RSIIndicator(close=self.df["Close"], window=14).rsi()
        
        # MACD (Moving Average Convergence Divergence): trend-following momentum indicator
        macd_indicator = ta.trend.MACD(
            close=self.df["Close"],
            window_fast=12,
            window_slow=26,
            window_sign=9
        )
        self.df["macd"] = macd_indicator.macd()
        self.df["macd_signal"] = macd_indicator.macd_signal()
        self.df["macd_hist"] = macd_indicator.macd_diff()  # Histogram = MACD - Signal line

        # Bollinger Bands: volatility-based envelope around price
        bb_indicator = ta.volatility.BollingerBands(
            close=self.df["Close"],
            window=20,
            window_dev=2
        )
        self.df["BBU"] = bb_indicator.bollinger_hband()  # Upper band
        self.df["BBL"] = bb_indicator.bollinger_lband()  # Lower band
        band_width = self.df["BBU"] - self.df["BBL"]
        bb_percent = (self.df["Close"] - self.df["BBL"]) / band_width
        self.df["bb_percent"] = np.clip(bb_percent, 0, 1).fillna(0.5)  # Normalized position within bands [0,1]

        # Simple Moving Averages (SMA): short-term (10) and long-term (50) trends
        self.df["sma_short"] = self.df["Close"].rolling(window=10).mean()
        self.df["sma_long"] = self.df["Close"].rolling(window=50).mean()

        # Volume Z-score: measures volume deviation from 20-period mean (clipped to [-5,5])
        volume_roll = self.df["Volume"].rolling(window=20)
        self.df["volume_mean"] = volume_roll.mean()
        self.df["volume_std"] = volume_roll.std().replace(0, np.nan)
        self.df["volume_z"] = (self.df["Volume"] - self.df["volume_mean"]) / self.df["volume_std"]
        self.df["volume_z"] = np.clip(self.df["volume_z"], -5, 5).fillna(0)

        # --- Log Returns (safe version) ---
        # Log returns over multiple time horizons (1, 5, 15, 60 steps)
        close_shifted_1 = self.df['Close'].shift(1)
        close_shifted_5 = self.df['Close'].shift(5)
        close_shifted_15 = self.df['Close'].shift(15)
        close_shifted_60 = self.df['Close'].shift(60)

        # Prevent division by zero by replacing zeros with tiny epsilon
        close_shifted_1 = close_shifted_1.replace(0, 1e-8).fillna(1e-8)
        close_shifted_5 = close_shifted_5.replace(0, 1e-8).fillna(1e-8)
        close_shifted_15 = close_shifted_15.replace(0, 1e-8).fillna(1e-8)
        close_shifted_60 = close_shifted_60.replace(0, 1e-8).fillna(1e-8)

        self.df['log_return_1'] = np.log(self.df['Close'] / close_shifted_1).fillna(0)
        self.df['log_return_5'] = np.log(self.df['Close'] / close_shifted_5).fillna(0)
        self.df['log_return_15'] = np.log(self.df['Close'] / close_shifted_15).fillna(0)
        self.df['log_return_60'] = np.log(self.df['Close'] / close_shifted_60).fillna(0)

        # --- Price Relative to SMAs ---
        # Normalized deviation of price from short and long SMAs (as % difference)
        sma_short_safe = self.df['sma_short'].replace(0, 1e-8)
        sma_long_safe = self.df['sma_long'].replace(0, 1e-8)

        self.df['price_vs_sma10'] = (self.df['Close'] / sma_short_safe) - 1
        self.df['price_vs_sma50'] = (self.df['Close'] / sma_long_safe) - 1

        # --- Volatility ---
        # Rolling 10-step standard deviation of 1-step log returns
        self.df['volatility_10'] = self.df['log_return_1'].rolling(window=10).std().fillna(0)

        # --- Time Features ---
        # Extract hour and day-of-week from timestamp
        self.df['hour'] = self.df['Open time'].dt.hour
        self.df['day_of_week'] = self.df['Open time'].dt.dayofweek

        # Encode cyclical time features using sine/cosine to preserve continuity
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)

        # --- SYMBOLIC LOGIC FEATURES (6 NEW) ---
        # These are handcrafted binary/weighted signals based on market conditions:
        # 1. Trend confidence: short & medium returns positive AND RSI not overbought
        self.df['trend_confidence'] = (
            (self.df['log_return_1'] > 0) &
            (self.df['log_return_5'] > 0) &
            (self.df['rsi'] < 70)
        ).astype(float) * 0.5

        # 2. Mean reversion signal: price above upper Bollinger Band AND RSI overbought
        self.df['mean_reversion'] = (
            (self.df['Close'] > self.df['BBU']) &
            (self.df['rsi'] > 70)
        ).astype(float) * 0.5

        # 3. Volume confirmation: high volume AND significant price move
        self.df['volume_confirmed'] = (
            (self.df['volume_z'] > 1.5) &
            (self.df['log_return_1'].abs() > 0.01)
        ).astype(float) * 0.3

        # 4. Breakout strength: price above long SMA, high volume, and positive MACD histogram
        self.df['breakout_strength'] = (
            (self.df['Close'] > self.df['sma_long']) &
            (self.df['volume_z'] > 1.0) &
            (self.df['macd_hist'] > 0)
        ).astype(float) * 0.4

        # 5. High volatility regime: current volatility > 2x 50-step average volatility
        self.df['high_volatility_regime'] = (
            self.df['volatility_10'] > 2 * self.df['volatility_10'].rolling(50).mean()
        ).astype(float) * 0.5

        # 6. Asia session bias: heuristic based on sine/cosine of hour (approx. 0–8 UTC)
        self.df['asia_session_bias'] = (
            (self.df['hour_sin'] > 0.8) &
            (self.df['hour_cos'] > 0.8)
        ).astype(float) * 0.3

        # --- Fill NaNs ---
        # Final cleanup: replace remaining NaNs and infinities, then drop any residual NaN rows
        self.df.fillna(0, inplace=True)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        if len(self.df) == 0:
            raise ValueError("❌ All data rows were dropped! Check your CSV.")

        print(f"📊 Dataset shape after cleaning: {self.df.shape}")

        # --- State Setup ---
        # Total number of observation features (must match _get_observation_raw)
        self.state_size = 24
        self.obs_rms = RunningMeanStd(shape=(self.state_size,))

        # Initialize trading state variables
        self.balance = self.initial_balance
        self.position = 0.0  # BTC held
        self.last_action = 0.0  # Previous normalized action [-1, 1]
        self.entry_price = None  # Price when position was opened
        self.holding_steps = 0  # How long current position has been held
        self.prev_position = 0.0
        self.returns_history = []  # Log returns history for Sharpe ratio
        self.max_equity = initial_balance

        self.reset()

    def _get_observation_raw(self, step):
        """
        Construct the raw (unnormalized) observation vector at a given time step.
        The observation has exactly 24 features, listed below by index:

        Index | Feature Name                | Description
        ------|-----------------------------|-----------------------------------------------
          0   | log_return_1                | 1-step log return
          1   | log_return_5                | 5-step log return
          2   | log_return_15               | 15-step log return
          3   | log_return_60               | 60-step log return
          4   | price_vs_sma10              | % deviation from 10-period SMA
          5   | price_vs_sma50              | % deviation from 50-period SMA
          6   | rsi                         | Relative Strength Index (0–100)
          7   | macd_hist                   | MACD histogram (momentum strength)
          8   | bb_percent                  | % position within Bollinger Bands [0,1]
          9   | volatility_10               | 10-step rolling volatility
         10   | volume_z                    | Volume Z-score (standardized volume)
         11   | equity_ratio                | Cash / Net Worth
         12   | position_ratio              | (Position * Price) / Net Worth
         13   | last_action                 | Previous action [-1, 1]
         14   | hour_sin                    | Sine of hour (cyclical encoding)
         15   | hour_cos                    | Cosine of hour (cyclical encoding)
         16   | day_sin                     | Sine of day-of-week
         17   | day_cos                     | Cosine of day-of-week
         18   | trend_confidence            | Weighted signal for trending market (0 or 0.5)
         19   | mean_reversion              | Weighted signal for overbought reversal (0 or 0.5)
         20   | volume_confirmed            | Weighted volume confirmation signal (0 or 0.3)
         21   | breakout_strength           | Weighted breakout signal (0 or 0.4)
         22   | high_volatility_regime      | Volatility regime flag (0 or 0.5)
         23   | asia_session_bias           | Heuristic Asia session flag (0 or 0.3)
        """
        frame = self.df.iloc[step]

        current_price = frame["Close"]
        net_worth = self.balance + self.position * current_price
        equity_ratio = self.balance / net_worth if net_worth > 0 else 0.0
        position_ratio = (self.position * current_price) / net_worth if net_worth > 0 else 0.0

        obs = np.array([
            frame['log_return_1'],
            frame['log_return_5'],
            frame['log_return_15'],
            frame['log_return_60'],
            frame['price_vs_sma10'],
            frame['price_vs_sma50'],
            frame['rsi'],
            frame['macd_hist'],
            frame['bb_percent'],
            frame['volatility_10'],
            frame['volume_z'],
            equity_ratio,
            position_ratio,
            self.last_action,
            frame['hour_sin'],
            frame['hour_cos'],
            frame['day_sin'],
            frame['day_cos'],
            frame['trend_confidence'],
            frame['mean_reversion'],
            frame['volume_confirmed'],
            frame['breakout_strength'],
            frame['high_volatility_regime'],
            frame['asia_session_bias'],
        ], dtype=np.float64)

        # Replace any remaining NaNs or infinities with 0 for safety
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _get_observation(self):
        # Get raw observation and optionally normalize it using running statistics
        obs_raw = self._get_observation_raw(self.current_step)
        if self.normalize_obs:
            obs_normalized = self.obs_rms.normalize(obs_raw)
            return obs_normalized.astype(np.float32)
        else:
            return obs_raw.astype(np.float32)

    def reset(self):
        """
        Reset the environment to initial state.
        - Restores balance, clears position, resets step counter.
        - Also resets the running normalization stats (note: this may affect training if done between episodes).
        - Returns initial observation.
        """
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0.0
        self.current_step = 0
        self.trades = []
        self.last_action = 0.0
        self.entry_price = None
        self.holding_steps = 0
        self.prev_position = 0.0
        self.returns_history = []
        self.max_equity = self.initial_balance
        self.obs_rms.reset()  # Resets normalization statistics
        return self._get_observation()

    def step(self, action):
        """
        Execute one trading step given an action.
        Action is a continuous value in [-1, 1], interpreted as target portfolio exposure to BTC.
        The environment:
          - Smooths action with previous action (70% new, 30% old)
          - Computes current exposure and desired change
          - Executes buy/sell with fee, respecting max trade size (50% of net worth)
          - Updates net worth using next price (note: uses next step's price immediately)
          - Computes reward as log return + bonus penalties/rewards
          - Applies drawdown penalty and Sharpe-based reward shaping
          - Terminates if net worth drops below $100 or end of data is reached
        """
        action = np.clip(action, -1.0, 1.0)
        action = 0.7 * action + 0.3 * self.last_action  # Action smoothing

        frame = self.df.iloc[self.current_step]
        price = frame["Close"]

        target_exposure = action
        current_value = self.position * price
        current_net_worth = self.balance + current_value
        current_exposure = current_value / current_net_worth if current_net_worth > 0 else 0.0
        delta_exposure = target_exposure - current_exposure
        max_trade_value = current_net_worth * 0.5  # Limit trade size to 50% of portfolio

        fee = 0.0
        # Execute buy if increasing exposure
        if delta_exposure > 0:
            invest_amount = min(delta_exposure * current_net_worth, max_trade_value)
            btc_to_buy = invest_amount / price
            cost = btc_to_buy * price
            fee = cost * self.transaction_fee
            self.position += btc_to_buy
            self.balance -= (cost + fee)
            self.trades.append(("buy", self.current_step, btc_to_buy, price, fee))
        # Execute sell if decreasing exposure
        elif delta_exposure < 0:
            sell_amount = min(-delta_exposure * current_net_worth, max_trade_value)
            btc_to_sell = sell_amount / price
            btc_to_sell = min(btc_to_sell, self.position)  # Cannot sell more than held
            revenue = btc_to_sell * price
            fee = revenue * self.transaction_fee
            self.position -= btc_to_sell
            self.balance += (revenue - fee)
            self.trades.append(("sell", self.current_step, btc_to_sell, price, fee))

        # Record previous net worth before advancing step
        prev_net_worth = self.net_worth
        self.current_step += 1
        # Clamp step to last valid index to avoid out-of-bounds (used when done=True)
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        # Get price at new step (this is the price used to value position after trade)
        price_now = self.df.iloc[self.current_step]["Close"]
        self.net_worth = self.balance + self.position * price_now

        # Compute log return of portfolio
        if prev_net_worth <= 1e-8:
            log_return = 0.0
        else:
            log_return = np.log(self.net_worth / prev_net_worth)

        reward = log_return * 1.0

        # --- Bonus/Penalty Rewards ---
        bonus_reward = 0.0
        # Reward for holding during strong trend
        if self.position > 0 and frame['trend_confidence'] > 0.4:
            bonus_reward += 0.05
        # Penalize aggressive trading during high volatility
        if abs(action) > 0.3 and frame['high_volatility_regime'] > 0.5:
            bonus_reward -= 0.05
        # Reward for holding during breakout
        if self.position > 0 and frame['breakout_strength'] > 0.3:
            bonus_reward += 0.03
        # Reward for stable actions (low churn)
        if self.position > 0 and abs(action - self.last_action) < 0.1:
            bonus_reward += 0.01

        reward += bonus_reward

        # --- Drawdown Penalty ---
        drawdown = (self.net_worth - self.max_equity) / self.max_equity if self.max_equity > 0 else 0.0
        if drawdown < -0.20:
            reward -= 0.5
        elif drawdown < -0.10:
            reward -= 0.2

        # --- Sharpe Ratio Reward Shaping ---
        if len(self.returns_history) >= 50:
            recent_returns = np.array(self.returns_history[-50:])
            sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)
            reward += np.clip(sharpe, -0.05, 0.1) * 0.1

        self.returns_history.append(log_return)

        # --- Position Tracking ---
        if self.position > 0 and self.prev_position == 0:
            # New long position opened
            self.entry_price = price_now
            self.holding_steps = 0
        elif self.position > 0:
            # Holding existing position
            self.holding_steps += 1
        elif self.position == 0 and self.prev_position > 0:
            # Position closed
            self.entry_price = None
            self.holding_steps = 0

        self.prev_position = self.position

        # Update max equity for drawdown calculation
        if self.current_step == 0:
            self.max_equity = self.initial_balance
        else:
            self.max_equity = max(self.max_equity, self.net_worth)

        # Check termination conditions
        done = self.current_step >= len(self.df) - 1

        # Catastrophic loss: stop if net worth drops too low
        if self.net_worth <= 100:
            done = True
            reward = -10.0

        self.last_action = action
        next_obs = self._get_observation()

        # Return (observation, reward, done, info)
        return next_obs, float(reward), bool(done), {}