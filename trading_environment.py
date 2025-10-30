
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
        self.clip_rewards = clip_rewards
        self.reward_function = default_reward_block

        # --- Technical Indicators ---
        # Existing indicators (unchanged)
        self.df["rsi"] = ta.momentum.RSIIndicator(close=self.df["Close"], window=14).rsi()
        
        macd_indicator = ta.trend.MACD(
            close=self.df["Close"],
            window_fast=12,
            window_slow=26,
            window_sign=9
        )
        self.df["macd"] = macd_indicator.macd()
        self.df["macd_signal"] = macd_indicator.macd_signal()
        self.df["macd_hist"] = macd_indicator.macd_diff()

        bb_indicator = ta.volatility.BollingerBands(
            close=self.df["Close"],
            window=20,
            window_dev=2
        )
        self.df["BBU"] = bb_indicator.bollinger_hband()
        self.df["BBL"] = bb_indicator.bollinger_lband()
        band_width = self.df["BBU"] - self.df["BBL"]
        bb_percent = (self.df["Close"] - self.df["BBL"]) / band_width
        self.df["bb_percent"] = np.clip(bb_percent, 0, 1).fillna(0.5)

        self.df["sma_short"] = self.df["Close"].rolling(window=10).mean()
        self.df["sma_long"] = self.df["Close"].rolling(window=50).mean()

        volume_roll = self.df["Volume"].rolling(window=20)
        self.df["volume_mean"] = volume_roll.mean()
        self.df["volume_std"] = volume_roll.std().replace(0, np.nan)
        self.df["volume_z"] = (self.df["Volume"] - self.df["volume_mean"]) / self.df["volume_std"]
        self.df["volume_z"] = np.clip(self.df["volume_z"], -5, 5).fillna(0)

        # --- EXISTING: 6 CRYPTO INDICATORS ---
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        cumulative_vp = (typical_price * self.df['Volume']).cumsum()
        cumulative_volume = self.df['Volume'].cumsum()
        self.df['vwap'] = cumulative_vp / cumulative_volume
        self.df['price_vs_vwap'] = (self.df['Close'] / self.df['vwap']) - 1

        atr_indicator = ta.volatility.AverageTrueRange(
            high=self.df['High'], 
            low=self.df['Low'], 
            close=self.df['Close'], 
            window=14
        )
        self.df['atr'] = atr_indicator.average_true_range()
        self.df['atr_percent'] = (self.df['atr'] / self.df['Close']) * 100

        stoch_rsi_indicator = ta.momentum.StochRSIIndicator(
            close=self.df['Close'], 
            window=14, 
            smooth1=3, 
            smooth2=3
        )
        self.df['stoch_rsi'] = stoch_rsi_indicator.stochrsi()

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
        
        self.df['ichimoku_cloud_bullish'] = (
            (self.df['Close'] > self.df['ichimoku_leading_a']) & 
            (self.df['Close'] > self.df['ichimoku_leading_b'])
        ).astype(float)

        price_movement = self.df['Close'] - self.df['Open']
        normalized_movement = price_movement / (self.df['High'] - self.df['Low']).replace(0, 1e-8)
        volume_weighted = normalized_movement * self.df['volume_z']
        self.df['order_imbalance_proxy'] = np.clip(volume_weighted, -1, 1).fillna(0)

        recent_high = self.df['High'].rolling(window=50).max()
        recent_low = self.df['Low'].rolling(window=50).min()
        price_position = (self.df['Close'] - recent_low) / (recent_high - recent_low).replace(0, 1e-8)
        liquidation_risk = self.df['atr_percent'] * np.abs(price_position - 0.5) * 2
        self.df['liquidation_risk'] = np.clip(liquidation_risk / 10, 0, 1).fillna(0)

        # --- NEW: 20 ADVANCED INDICATORS ---

        # 1. REGIME DETECTION (4 indicators)
        # Market Regime Classification
        returns = self.df['Close'].pct_change().fillna(0)
        self.df['volatility_regime'] = returns.rolling(20).std().fillna(0)
        
        # Trend Regime (ADX)
        adx_indicator = ta.trend.ADXIndicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=14)
        self.df['adx'] = adx_indicator.adx()
        self.df['trend_regime'] = (self.df['adx'] > 25).astype(float)
        
        # Momentum Regime (Composite)
        williams_r = ta.momentum.WilliamsRIndicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], lbp=14).williams_r()
        self.df['momentum_regime'] = (self.df['rsi'] + self.df['stoch_rsi'] + (williams_r + 100) / 2) / 3
        
        # Cycle Regime (Simple rolling dominant cycle)
        def calculate_dominant_cycle(x):
            if len(x) < 50:
                return 25
            try:
                fft = np.fft.fft(x - np.mean(x))
                magnitudes = np.abs(fft[:len(fft)//2])
                return np.argmax(magnitudes[1:]) + 1  # Skip DC component
            except:
                return 25
                
        self.df['cycle_regime'] = self.df['Close'].rolling(50).apply(calculate_dominant_cycle, raw=False).fillna(25) / 50

        # 2. ADVANCED MOMENTUM (5 indicators)
        # Chande Momentum Oscillator (Manual Implementation)
        def calculate_cmo(close, window=14):
            gains = np.where(close.diff() > 0, close.diff(), 0)
            losses = np.where(close.diff() < 0, -close.diff(), 0)
            sum_gains = pd.Series(gains).rolling(window=window).sum()
            sum_losses = pd.Series(losses).rolling(window=window).sum()
            cmo = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
            return cmo.fillna(0)
        
        self.df['cmo'] = calculate_cmo(self.df['Close'], window=14)
        
        # Know Sure Thing (KST) - Simplified
        roc_10 = ta.momentum.ROCIndicator(close=self.df['Close'], window=10).roc()
        roc_15 = ta.momentum.ROCIndicator(close=self.df['Close'], window=15).roc()
        roc_20 = ta.momentum.ROCIndicator(close=self.df['Close'], window=20).roc()
        roc_30 = ta.momentum.ROCIndicator(close=self.df['Close'], window=30).roc()
        self.df['kst'] = (roc_10.rolling(10).mean() + 
                          roc_15.rolling(10).mean() * 2 + 
                          roc_20.rolling(10).mean() * 3 + 
                          roc_30.rolling(15).mean() * 4).fillna(0)
        
        # True Strength Index (Manual Implementation)
        def calculate_tsi(close, slow=25, fast=13):
            diff = close.diff()
            double_smoothed = diff.ewm(span=fast).mean().ewm(span=slow).mean()
            abs_diff_smoothed = diff.abs().ewm(span=fast).mean().ewm(span=slow).mean()
            tsi = 100 * (double_smoothed / abs_diff_smoothed)
            return tsi.fillna(0)
        
        self.df['tsi'] = calculate_tsi(self.df['Close'])
        
        # Elder-Ray Index
        ema_13 = ta.trend.EMAIndicator(close=self.df['Close'], window=13).ema_indicator()
        self.df['bull_power'] = (self.df['High'] - ema_13).fillna(0)
        self.df['bear_power'] = (self.df['Low'] - ema_13).fillna(0)
        
        # Vortex Indicator
        vi_indicator = ta.trend.VortexIndicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=14)
        self.df['vi_plus'] = vi_indicator.vortex_indicator_pos().fillna(0)
        self.df['vi_minus'] = vi_indicator.vortex_indicator_neg().fillna(0)

        # 3. SOPHISTICATED VOLUME (4 indicators)
        # Volume Price Confirmation
        vpc = self.df['Volume'] * (self.df['Close'] - self.df['Close'].shift(1)) / self.df['Close'].shift(1)
        self.df['vpci'] = vpc.rolling(20).mean().fillna(0)
        
        # Ease of Movement
        eom_indicator = ta.volume.EaseOfMovementIndicator(high=self.df['High'], low=self.df['Low'], volume=self.df['Volume'], window=14)
        self.df['eom'] = eom_indicator.ease_of_movement().fillna(0)
        
        # Volume-Weighted Momentum
        self.df['price_volume_trend'] = (self.df['Close'].diff() / self.df['Close'].shift(1)) * self.df['Volume']
        self.df['price_volume_trend'] = self.df['price_volume_trend'].fillna(0)
        
        # Accumulation/Distribution Line
        adl_indicator = ta.volume.AccDistIndexIndicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], volume=self.df['Volume'])
        self.df['adl'] = adl_indicator.acc_dist_index().fillna(0)

        # 4. ADVANCED VOLATILITY & RISK (4 indicators)
        # Donchian Channel Width
        donchian_high = self.df['High'].rolling(20).max()
        donchian_low = self.df['Low'].rolling(20).min()
        self.df['donchian_width'] = ((donchian_high - donchian_low) / self.df['Close']).fillna(0)
        
        # Historical VaR
        self.df['var_95'] = returns.rolling(100).quantile(0.05).fillna(0)
        
        # Conditional VaR (Manual Implementation)
        def calculate_cvar(returns_series):
            if len(returns_series) < 100:
                return 0
            var_level = returns_series.quantile(0.05)
            cvar = returns_series[returns_series <= var_level].mean()
            return cvar if not np.isnan(cvar) else 0
            
        self.df['cvar_95'] = returns.rolling(100).apply(calculate_cvar, raw=False).fillna(0)
        
        # Volatility Smile Proxy
        atr_short = ta.volatility.AverageTrueRange(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=5).average_true_range()
        atr_long = ta.volatility.AverageTrueRange(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=20).average_true_range()
        self.df['atr_ratio'] = (atr_short / atr_long).fillna(1)

        # 5. MARKET MICROSTRUCTURE (3 indicators)
        # Volume Delta
        buy_volume = self.df['Volume'] * (self.df['Close'] - self.df['Open']).apply(lambda x: 1 if x > 0 else 0)
        sell_volume = self.df['Volume'] * (self.df['Close'] - self.df['Open']).apply(lambda x: 1 if x < 0 else 0)
        self.df['volume_delta'] = ((buy_volume - sell_volume) / self.df['Volume'].replace(0, 1)).fillna(0)
        
        # Volume Clustering
        large_trade_threshold = self.df['Volume'].rolling(50).quantile(0.8)
        self.df['large_trade_ratio'] = (self.df['Volume'] > large_trade_threshold).rolling(20).mean().fillna(0)
        
        # Price-Volume Correlation
        self.df['price_volume_corr'] = self.df['Close'].rolling(20).corr(self.df['Volume']).fillna(0)

        # --- EXISTING FEATURES (unchanged) ---
        close_shifted_1 = self.df['Close'].shift(1).replace(0, 1e-8).fillna(1e-8)
        close_shifted_5 = self.df['Close'].shift(5).replace(0, 1e-8).fillna(1e-8)
        close_shifted_15 = self.df['Close'].shift(15).replace(0, 1e-8).fillna(1e-8)
        close_shifted_60 = self.df['Close'].shift(60).replace(0, 1e-8).fillna(1e-8)

        self.df['log_return_1'] = np.log(self.df['Close'] / close_shifted_1).fillna(0)
        self.df['log_return_5'] = np.log(self.df['Close'] / close_shifted_5).fillna(0)
        self.df['log_return_15'] = np.log(self.df['Close'] / close_shifted_15).fillna(0)
        self.df['log_return_60'] = np.log(self.df['Close'] / close_shifted_60).fillna(0)

        sma_short_safe = self.df['sma_short'].replace(0, 1e-8)
        sma_long_safe = self.df['sma_long'].replace(0, 1e-8)
        self.df['price_vs_sma10'] = (self.df['Close'] / sma_short_safe) - 1
        self.df['price_vs_sma50'] = (self.df['Close'] / sma_long_safe) - 1

        self.df['volatility_10'] = self.df['log_return_1'].rolling(window=10).std().fillna(0)

        self.df['hour'] = self.df['Open time'].dt.hour
        self.df['day_of_week'] = self.df['Open time'].dt.dayofweek
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)

        self.df['trend_confidence'] = (
            (self.df['log_return_1'] > 0) & (self.df['log_return_5'] > 0) & (self.df['rsi'] < 70)
        ).astype(float) * 0.5

        self.df['mean_reversion'] = (
            (self.df['Close'] > self.df['BBU']) & (self.df['rsi'] > 70)
        ).astype(float) * 0.5

        self.df['volume_confirmed'] = (
            (self.df['volume_z'] > 1.5) & (self.df['log_return_1'].abs() > 0.01)
        ).astype(float) * 0.3

        self.df['breakout_strength'] = (
            (self.df['Close'] > self.df['sma_long']) & (self.df['volume_z'] > 1.0) & (self.df['macd_hist'] > 0)
        ).astype(float) * 0.4

        self.df['high_volatility_regime'] = (
            self.df['volatility_10'] > 2 * self.df['volatility_10'].rolling(50).mean()
        ).astype(float) * 0.5

        self.df['asia_session_bias'] = (
            (self.df['hour_sin'] > 0.8) & (self.df['hour_cos'] > 0.8)
        ).astype(float) * 0.3

        # --- Fill NaNs ---
        self.df.fillna(0, inplace=True)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        if len(self.df) == 0:
            raise ValueError("❌ All data rows were dropped! Check your CSV.")

        print(f"📊 Dataset shape after cleaning: {self.df.shape}")

        # --- State Setup ---
        self.state_size = 50  # UPDATED: 30 previous + 20 new indicators
        self.obs_rms = RunningMeanStd(shape=(self.state_size,))

        # Initialize trading state variables (unchanged)
        self.balance = self.initial_balance
        self.position = 0.0
        self.last_action = 0.0
        self.entry_price = None
        self.holding_steps = 0
        self.prev_position = 0.0
        self.returns_history = []
        self.max_equity = initial_balance

        self.reset()

    # KEEP THE REST OF THE CODE EXACTLY THE SAME (including _get_observation_raw with table comments)
    def _get_observation_raw(self, step):
        """
        Construct the raw (unnormalized) observation vector at a given time step.
        The observation has exactly 50 features, organized as follows:

        =============================================================================
        CATEGORY           | INDEX RANGE | COUNT | DESCRIPTION
        =============================================================================
        PRICE & RETURNS    |   0 -  5    |   6   | Multi-timeframe log returns & SMA deviations
        TECHNICAL INDICATORS | 6 - 10    |   5   | RSI, MACD, Bollinger Bands, Volatility, Volume
        PORTFOLIO STATE    |  11 - 13    |   3   | Equity ratio, position ratio, last action
        TIME FEATURES      |  14 - 17    |   4   | Cyclical hour/day encodings (sine/cosine)
        SYMBOLIC LOGIC     |  18 - 23    |   6   | Expert-crafted trading signals
        CRYPTO INDICATORS  |  24 - 29    |   6   | VWAP, ATR%, Stoch RSI, Ichimoku, Order Imbalance, Liquidation Risk
        ADVANCED INDICATORS|  30 - 49    |  20   | Regime detection, advanced momentum, volume, volatility, microstructure
        =============================================================================
        """
        frame = self.df.iloc[step]

        current_price = frame["Close"]
        net_worth = self.balance + self.position * current_price
        equity_ratio = self.balance / net_worth if net_worth > 0 else 0.0
        position_ratio = (self.position * current_price) / net_worth if net_worth > 0 else 0.0

        obs = np.array([
            # ==================== ORIGINAL 30 FEATURES (0-29) ====================
            frame['log_return_1'], frame['log_return_5'], frame['log_return_15'], frame['log_return_60'],
            frame['price_vs_sma10'], frame['price_vs_sma50'], frame['rsi'], frame['macd_hist'],
            frame['bb_percent'], frame['volatility_10'], frame['volume_z'], equity_ratio,
            position_ratio, self.last_action, frame['hour_sin'], frame['hour_cos'],
            frame['day_sin'], frame['day_cos'], frame['trend_confidence'], frame['mean_reversion'],
            frame['volume_confirmed'], frame['breakout_strength'], frame['high_volatility_regime'],
            frame['asia_session_bias'], frame['price_vs_vwap'], frame['atr_percent'],
            frame['stoch_rsi'], frame['ichimoku_cloud_bullish'], frame['order_imbalance_proxy'],
            frame['liquidation_risk'],
            
            # ==================== NEW 20 ADVANCED FEATURES (30-49) ====================
            frame['volatility_regime'], frame['trend_regime'], frame['momentum_regime'], frame['cycle_regime'],
            frame['cmo'], frame['kst'], frame['tsi'], frame['bull_power'], frame['bear_power'],
            frame['vi_plus'], frame['vi_minus'], frame['vpci'], frame['eom'], frame['price_volume_trend'],
            frame['adl'], frame['donchian_width'], frame['var_95'], frame['cvar_95'], frame['atr_ratio'],
            frame['volume_delta']
        ], dtype=np.float64)

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _get_observation(self):
        obs_raw = self._get_observation_raw(self.current_step)
        if self.normalize_obs:
            obs_normalized = self.obs_rms.normalize(obs_raw)
            return obs_normalized.astype(np.float32)
        else:
            return obs_raw.astype(np.float32)

    def reset(self):
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
        
        if hasattr(self, 'obs_rms') and self.obs_rms.mean.shape == (self.state_size,):
            self.obs_rms.reset()
        else:
            self.obs_rms = RunningMeanStd(shape=(self.state_size,))
            
        return self._get_observation()

    def step(self, action):
        # EXACTLY THE SAME AS BEFORE
        action = np.clip(action, -1.0, 1.0)
        action = 0.7 * action + 0.3 * self.last_action

        frame = self.df.iloc[self.current_step]
        price = frame["Close"]

        target_exposure = action
        current_value = self.position * price
        current_net_worth = self.balance + current_value
        current_exposure = current_value / current_net_worth if current_net_worth > 0 else 0.0
        delta_exposure = target_exposure - current_exposure
        max_trade_value = current_net_worth * 0.5

        fee = 0.0
        if delta_exposure > 0:
            invest_amount = min(delta_exposure * current_net_worth, max_trade_value)
            btc_to_buy = invest_amount / price
            cost = btc_to_buy * price
            fee = cost * self.transaction_fee
            self.position += btc_to_buy
            self.balance -= (cost + fee)
            self.trades.append(("buy", self.current_step, btc_to_buy, price, fee))
        elif delta_exposure < 0:
            sell_amount = min(-delta_exposure * current_net_worth, max_trade_value)
            btc_to_sell = sell_amount / price
            btc_to_sell = min(btc_to_sell, self.position)
            revenue = btc_to_sell * price
            fee = revenue * self.transaction_fee
            self.position -= btc_to_sell
            self.balance += (revenue - fee)
            self.trades.append(("sell", self.current_step, btc_to_sell, price, fee))

        prev_net_worth = self.net_worth
        self.current_step += 1
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        price_now = self.df.iloc[self.current_step]["Close"]
        self.net_worth = self.balance + self.position * price_now

        if prev_net_worth <= 1e-8:
            log_return = 0.0
        else:
            log_return = np.log(self.net_worth / prev_net_worth)

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

        self.returns_history.append(log_return)

        if self.position > 0 and self.prev_position == 0:
            self.entry_price = price_now
            self.holding_steps = 0
        elif self.position > 0:
            self.holding_steps += 1
        elif self.position == 0 and self.prev_position > 0:
            self.entry_price = None
            self.holding_steps = 0

        self.prev_position = self.position

        if self.current_step == 0:
            self.max_equity = self.initial_balance
        else:
            self.max_equity = max(self.max_equity, self.net_worth)

        done = self.current_step >= len(self.df) - 1

        if self.net_worth <= 100:
            done = True
            reward = -10.0

        self.last_action = action
        next_obs = self._get_observation()

        return next_obs, float(reward), bool(done), {}