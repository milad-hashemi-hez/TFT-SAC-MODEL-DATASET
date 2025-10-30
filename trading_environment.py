
# trading_environment.py

import numpy as np
import pandas as pd
import ta
from reward_block import default_reward_block

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
        self.reward_function = default_reward_block  # reward_block related

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

        # --- NEW: ESSENTIAL CRYPTO INDICATORS (6 ADDED) ---
        
        # 1. VWAP (Volume Weighted Average Price)
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        cumulative_vp = (typical_price * self.df['Volume']).cumsum()
        cumulative_volume = self.df['Volume'].cumsum()
        self.df['vwap'] = cumulative_vp / cumulative_volume
        self.df['price_vs_vwap'] = (self.df['Close'] / self.df['vwap']) - 1  # % deviation from VWAP

        # 2. ATR% (Normalized Average True Range)
        atr_indicator = ta.volatility.AverageTrueRange(
            high=self.df['High'], 
            low=self.df['Low'], 
            close=self.df['Close'], 
            window=14
        )
        self.df['atr'] = atr_indicator.average_true_range()
        self.df['atr_percent'] = (self.df['atr'] / self.df['Close']) * 100  # Normalized ATR

        # 3. Stochastic RSI (More sensitive momentum)
        stoch_rsi_indicator = ta.momentum.StochRSIIndicator(
            close=self.df['Close'], 
            window=14, 
            smooth1=3, 
            smooth2=3
        )
        self.df['stoch_rsi'] = stoch_rsi_indicator.stochrsi()

        # 4. Ichimoku Cloud Components
        ichimoku_indicator = ta.trend.IchimokuIndicator(
            high=self.df['High'],
            low=self.df['Low'],
            window1=9,
            window2=26,
            window3=52
        )
        self.df['ichimoku_conversion'] = ichimoku_indicator.ichimoku_conversion_line()
        self.df['ichimoku_base'] = ichimoku_indicator.ichimoku_base_line()
        self.df['ichimoku_leading_a'] = ichimoku_indicator.ichimoku_a()
        self.df['ichimoku_leading_b'] = ichimoku_indicator.ichimoku_b()
        
        # Ichimoku Cloud Position (simplified)
        self.df['ichimoku_cloud_bullish'] = (
            (self.df['Close'] > self.df['ichimoku_leading_a']) & 
            (self.df['Close'] > self.df['ichimoku_leading_b'])
        ).astype(float)

        # 5. Order Book Imbalance (Simulated - using OHLCV patterns)
        # This is a proxy since we don't have real order book data
        price_movement = self.df['Close'] - self.df['Open']
        normalized_movement = price_movement / (self.df['High'] - self.df['Low']).replace(0, 1e-8)
        volume_weighted = normalized_movement * self.df['volume_z']
        self.df['order_imbalance_proxy'] = np.clip(volume_weighted, -1, 1).fillna(0)

        # 6. Liquidation Levels Proxy (using volatility and price extremes)
        # Simulates liquidation pressure zones
        recent_high = self.df['High'].rolling(window=50).max()
        recent_low = self.df['Low'].rolling(window=50).min()
        price_position = (self.df['Close'] - recent_low) / (recent_high - recent_low).replace(0, 1e-8)
        # Higher volatility + extreme price positions = higher liquidation risk
        liquidation_risk = self.df['atr_percent'] * np.abs(price_position - 0.5) * 2
        self.df['liquidation_risk'] = np.clip(liquidation_risk / 10, 0, 1).fillna(0)  # Normalized to [0,1]

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
        # Total number of observation features (increased from 24 to 30)
        self.state_size = 30  # UPDATED: 24 original + 6 new indicators
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
        The observation now has 30 features (24 original + 6 new crypto indicators).
        
        Original 24 features (0-23):
        [0-5] Price & Returns, [6-10] Technical Indicators, [11-13] Portfolio State,
        [14-17] Time Features, [18-23] Symbolic Logic Features
        
        NEW 6 features (24-29):
        Index | Feature Name           | Description
        ------|------------------------|-----------------------------------------------
         24   | price_vs_vwap         | % deviation from VWAP
         25   | atr_percent           | Normalized ATR (% of price)
         26   | stoch_rsi             | Stochastic RSI (more sensitive momentum)
         27   | ichimoku_cloud_bullish| Ichimoku cloud position (0 or 1)
         28   | order_imbalance_proxy | Simulated order book imbalance
         29   | liquidation_risk      | Liquidation risk estimate [0,1]
        """
        frame = self.df.iloc[step]

        current_price = frame["Close"]
        net_worth = self.balance + self.position * current_price
        equity_ratio = self.balance / net_worth if net_worth > 0 else 0.0
        position_ratio = (self.position * current_price) / net_worth if net_worth > 0 else 0.0

        obs = np.array([
            # Original 24 features (0-23)
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
            
            # NEW: 6 essential crypto indicators (24-29)
            frame['price_vs_vwap'],           # 24
            frame['atr_percent'],             # 25
            frame['stoch_rsi'],               # 26
            frame['ichimoku_cloud_bullish'],  # 27
            frame['order_imbalance_proxy'],   # 28
            frame['liquidation_risk']         # 29
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
        
        # FIX: Only reset normalization stats if they have the correct shape
        if hasattr(self, 'obs_rms') and self.obs_rms.mean.shape == (self.state_size,):
            self.obs_rms.reset()
        else:
            # Reinitialize if shape is wrong (safety check)
            self.obs_rms = RunningMeanStd(shape=(self.state_size,))
            
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

        ##############################################################################################
        ############################## Reward Block ##################################################
        ##############################################################################################    

        reward = self.reward_function(
            log_return=log_return,
            position=self.position,
            action=action,
            last_action=self.last_action,
            frame=frame,
            net_worth=self.net_worth,
            max_equity=self.max_equity,
            returns_history=self.returns_history
        )

        ##############################################################################################
        ##############################################################################################
        ##############################################################################################     

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