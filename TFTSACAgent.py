
# TFTSACAgent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from typing import Tuple, Optional

class TFTFeatureProcessor(nn.Module):
    """Process features for TFT - handles static, temporal, and known variables"""
    def __init__(self, state_size: int, hidden_size: int):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        
        # Linear transformation for input features
        self.feature_projection = nn.Linear(state_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, state_size)
        projected = self.feature_projection(x)  # (batch_size, seq_len, hidden_size)
        normalized = self.layer_norm(projected)
        return F.relu(normalized)

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for feature processing"""
    def __init__(self, input_size: int, hidden_size: int = None):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
            
        self.linear = nn.Linear(input_size, hidden_size * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        gate, value = out.chunk(2, dim=-1)
        return F.glu(torch.cat([value, gate], dim=-1), dim=-1)

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature importance"""
    def __init__(self, input_size: int, hidden_size: int, num_inputs: int):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        
        # Squeeze network to calculate variable selection weights
        self.selu = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, num_inputs),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, num_inputs, hidden_size)
        batch_size, seq_len, num_inputs, hidden_size = x.shape
        
        # Calculate selection weights
        flat_x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, num_inputs * hidden_size)
        weights = self.selu(flat_x)  # (batch_size, seq_len, num_inputs)
        
        # Reshape weights and apply to inputs
        weights = weights.unsqueeze(-1).expand(-1, -1, -1, hidden_size)  # (batch_size, seq_len, num_inputs, hidden_size)
        weighted_inputs = x * weights
        
        # Sum across inputs
        output = weighted_inputs.sum(dim=2)  # (batch_size, seq_len, hidden_size)
        
        return output, weights

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention for temporal relationships"""
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(output)
        
        return output

class TemporalFusionEncoder(nn.Module):
    """Main TFT Encoder with attention and gating mechanisms"""
    def __init__(self, state_size: int, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Feature processing
        self.feature_processor = TFTFeatureProcessor(state_size, hidden_size)
        
        # Variable selection network
        self.variable_selection = VariableSelectionNetwork(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_inputs=1
        )
        
        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        
        # Gated residual connections
        self.gate1 = GatedLinearUnit(hidden_size)
        self.gate2 = GatedLinearUnit(hidden_size)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # Process features
        processed_features = self.feature_processor(x)
        
        # Variable selection (reshape for VSN)
        reshaped_features = processed_features.unsqueeze(2)  # (batch_size, seq_len, 1, hidden_size)
        selected_features, _ = self.variable_selection(reshaped_features)
        
        # Self-attention
        attention_output = self.attention(selected_features)
        
        # First residual connection with gating
        gate1_output = self.gate1(attention_output)
        residual1 = self.norm1(selected_features + gate1_output)
        
        # Second processing layer (feed-forward)
        ff_output = F.relu(residual1)
        gate2_output = self.gate2(ff_output)
        
        # Second residual connection
        output = self.norm2(residual1 + gate2_output)
        
        return output

# =============================================================================
# FORECAST HEAD IMPLEMENTATION - STEP 1
# =============================================================================
class TFTForecastHead(nn.Module):
    """TFT Forecast Head for price prediction with uncertainty"""
    def __init__(self, hidden_size: int, num_quantiles: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_quantiles = num_quantiles
        
        # Point prediction (mean)
        self.point_proj = nn.Linear(hidden_size, 1)
        
        # Quantile predictions for uncertainty
        self.quantile_proj = nn.Linear(hidden_size, num_quantiles)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, encoded_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoded_seq shape: (batch_size, seq_len, hidden_size)
        
        # Use the last time step for forecasting
        last_hidden = encoded_seq[:, -1, :]  # (batch_size, hidden_size)
        normalized = self.layer_norm(last_hidden)
        
        # Point prediction (mean price)
        point_pred = self.point_proj(normalized)  # (batch_size, 1)
        
        # Quantile predictions (uncertainty intervals)
        quantile_pred = self.quantile_proj(normalized)  # (batch_size, num_quantiles)
        
        return point_pred, quantile_pred

class TFTSACAgent:
    def __init__(self, state_size=24, action_size=1, lr=3e-4, gamma=0.99, alpha=0.2, 
                 tau=0.005, batch_size=128, seq_len=20, hidden_size=128, num_heads=8):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # AUTOMATIC ENTROPY TUNING
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp().item()
        self.target_entropy = -float(action_size)

        # TFT Encoder (replacing LSTM)
        self.tft_encoder = TemporalFusionEncoder(
            state_size=state_size, 
            hidden_size=hidden_size, 
            num_heads=num_heads
        ).to(self.device)

        # =============================================================================
        # FORECAST HEAD INITIALIZATION - STEP 2
        # =============================================================================
        self.forecast_head = TFTForecastHead(
            hidden_size=hidden_size,
            num_quantiles=3  # Predict 0.1, 0.5, 0.9 quantiles
        ).to(self.device)

        # Forecast optimizer
        self.forecast_optimizer = optim.Adam(self.forecast_head.parameters(), lr=lr)
        # =============================================================================

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * action_size)
        ).to(self.device)

        # Twin Critic Networks - UPDATED FOR FORECAST FEATURES
        self.critic1 = nn.Sequential(
            nn.Linear(hidden_size + action_size + 2, 128),  # +2 for forecast features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.critic2 = nn.Sequential(
            nn.Linear(hidden_size + action_size + 2, 128),  # +2 for forecast features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        # Target Networks
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self._freeze_params(self.critic1_target)
        self._freeze_params(self.critic2_target)

        # Optimizers
        self.actor_optimizer = optim.Adam(
            list(self.tft_encoder.parameters()) + list(self.actor.parameters()), 
            lr=lr
        )
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.memory = []
        self.max_memory = 100000

    def _freeze_params(self, model):
        for param in model.parameters():
            param.requires_grad = False

    # =============================================================================
    # FORECAST LOSS FUNCTION - STEP 2
    # =============================================================================
    def _quantile_loss(self, target: torch.Tensor, quantile_pred: torch.Tensor, 
                       quantiles: torch.Tensor = None) -> torch.Tensor:
        """Calculate quantile loss for uncertainty estimation"""
        if quantiles is None:
            quantiles = torch.tensor([0.1, 0.5, 0.9], device=self.device)
        
        # target shape: (batch_size, 1)
        # quantile_pred shape: (batch_size, num_quantiles)
        errors = target.unsqueeze(-1) - quantile_pred.unsqueeze(1)  # (batch_size, 1, num_quantiles)
        
        # Quantile loss calculation
        losses = torch.max((quantiles - 1) * errors, quantiles * errors)
        return losses.mean()
    # =============================================================================

    def act(self, state, evaluate=False):
        # state shape: (seq_len, state_size)
        state_batch = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, seq_len, state_size)
        
        with torch.no_grad():
            # Encode sequence using TFT
            encoded_seq = self.tft_encoder(state_batch)  # (1, seq_len, hidden_size)
            # Take the last time step as the representation
            encoded_features = encoded_seq[:, -1, :]  # (1, hidden_size)

            mean_logstd = self.actor(encoded_features)
            mean = mean_logstd[:, :self.action_size]
            log_std = mean_logstd[:, self.action_size:]
            log_std = torch.clamp(log_std, -5, 2)
            std = log_std.exp()

            if evaluate:
                raw_action = mean
                action = F.tanh(raw_action)
                log_prob = None
            else:
                normal = Normal(mean, std)
                raw_action = normal.rsample()
                action = F.tanh(raw_action)

                action_detached = action.detach()
                tanh_correction = torch.log(1 - action_detached.pow(2) + 1e-6)
                log_prob_raw = normal.log_prob(raw_action)
                log_prob = log_prob_raw - tanh_correction
                log_prob = log_prob.sum(-1, keepdim=True)

            return action.item(), log_prob.item() if not evaluate else None

    def remember(self, state_seq, action, reward, next_state_seq, done):
        self.memory.append((state_seq, action, reward, next_state_seq, done))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def update_alpha(self, log_prob):
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return None, None, None

        batch = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
        state_seqs, actions, rewards, next_state_seqs, dones = zip(*[self.memory[i] for i in batch])

        state_seqs = torch.FloatTensor(np.array(state_seqs)).to(self.device)  # (batch_size, seq_len, state_size)
        actions = torch.FloatTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_seqs = torch.FloatTensor(np.array(next_state_seqs)).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        # =============================================================================
        # FORECAST TARGET PREPARATION - STEP 3
        # =============================================================================
        # Use next step's close price as forecast target (simple approach)
        batch_size, seq_len, state_dim = state_seqs.shape
        forecast_targets = state_seqs[:, -1, 0].unsqueeze(1)  # Use Close price as target
        # =============================================================================

        # Encode sequences using TFT
        current_encoded = self.tft_encoder(state_seqs)  # (batch_size, seq_len, hidden_size)
        next_encoded = self.tft_encoder(next_state_seqs)  # (batch_size, seq_len, hidden_size)

        # =============================================================================
        # FORECAST PREDICTIONS - STEP 3
        # =============================================================================
        point_pred, quantile_pred = self.forecast_head(current_encoded)
        forecast_loss = self._quantile_loss(forecast_targets, quantile_pred)
        # =============================================================================

        # Take last time step representations
        current_features = current_encoded[:, -1, :]  # (batch_size, hidden_size)
        next_features = next_encoded[:, -1, :]  # (batch_size, hidden_size)

        # CRITIC LOSS - DETACH current_features for critic updates to avoid gradient conflicts
        current_features_detached = current_features.detach()  # Create detached copy for critics

        # =============================================================================
        # MODIFIED CRITIC INPUTS WITH FORECAST FEATURES - STEP 3
        # =============================================================================
        forecast_features = torch.cat([point_pred.detach(), quantile_pred.mean(dim=-1, keepdim=True).detach()], dim=-1)
        critic_input = torch.cat([current_features_detached, actions, forecast_features], dim=1)

        current_q1 = self.critic1(critic_input)
        current_q2 = self.critic2(critic_input)
        # =============================================================================

        # TARGET Q VALUES
        with torch.no_grad():
            next_mean_logstd = self.actor(next_features)
            next_mean = next_mean_logstd[:, :self.action_size]
            next_log_std = next_mean_logstd[:, self.action_size:]
            next_log_std = torch.clamp(next_log_std, -5, 2)
            next_std = next_log_std.exp()

            next_normal = Normal(next_mean, next_std)
            next_raw_action = next_normal.rsample()
            next_action = F.tanh(next_raw_action)

            next_action_detached = next_action.detach()
            next_tanh_correction = torch.log(1 - next_action_detached.pow(2) + 1e-6)
            next_log_prob_raw = next_normal.log_prob(next_raw_action)
            next_log_prob = next_log_prob_raw - next_tanh_correction
            next_log_prob = next_log_prob.sum(-1, keepdim=True)

            # Target critics also use forecast features
            next_forecast_features = torch.cat([point_pred.detach(), quantile_pred.mean(dim=-1, keepdim=True).detach()], dim=-1)
            next_critic_input = torch.cat([next_features, next_action, next_forecast_features], dim=1)
            
            target_q1 = self.critic1_target(next_critic_input)
            target_q2 = self.critic2_target(next_critic_input)
            min_target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones.float()) * (min_target_q - self.alpha * next_log_prob)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        # OPTIMIZE CRITICS
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)  # Keep graph for actor update
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward(retain_graph=True)  # Keep graph for actor update
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic2_optimizer.step()

        # =============================================================================
        # FORECAST HEAD OPTIMIZATION - STEP 3
        # =============================================================================
        self.forecast_optimizer.zero_grad()
        forecast_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.forecast_head.parameters(), max_norm=1.0)
        self.forecast_optimizer.step()
        # =============================================================================

        # ACTOR LOSS - Use original current_features (not detached) for actor update
        actor_mean_logstd = self.actor(current_features)  # Use original features
        actor_mean = actor_mean_logstd[:, :self.action_size]
        actor_log_std = actor_mean_logstd[:, self.action_size:]
        actor_log_std = torch.clamp(actor_log_std, -5, 2)
        actor_std = actor_log_std.exp()

        actor_normal = Normal(actor_mean, actor_std)
        actor_raw_action = actor_normal.rsample()
        actor_action = F.tanh(actor_raw_action)

        actor_action_detached = actor_action.detach()
        actor_tanh_correction = torch.log(1 - actor_action_detached.pow(2) + 1e-6)
        actor_log_prob_raw = actor_normal.log_prob(actor_raw_action)
        actor_log_prob = actor_log_prob_raw - actor_tanh_correction
        actor_log_prob = actor_log_prob.sum(-1, keepdim=True)

        with torch.no_grad():
            actor_forecast_features = torch.cat([point_pred.detach(), quantile_pred.mean(dim=-1, keepdim=True).detach()], dim=-1)
            actor_critic_input = torch.cat([current_features_detached, actor_action, actor_forecast_features], dim=1)
            
            q1_actor = self.critic1(actor_critic_input)
            q2_actor = self.critic2(actor_critic_input)
            min_q_actor = torch.min(q1_actor, q2_actor)

        actor_loss = (self.alpha * actor_log_prob - min_q_actor).mean()

        # OPTIMIZE ACTOR (including TFT encoder)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.tft_encoder.parameters()) + list(self.actor.parameters()), 
            max_norm=1.0
        )
        self.actor_optimizer.step()

        # UPDATE ENTROPY
        self.update_alpha(actor_log_prob)

        # UPDATE TARGET NETWORKS
        self._soft_update_target_networks()

        # =============================================================================
        # RETURN FORECAST LOSS IN RESULTS - STEP 3
        # =============================================================================
        return actor_loss.item(), (critic1_loss.item() + critic2_loss.item()) / 2, self.alpha, forecast_loss.item()
        # =============================================================================

    def _soft_update_target_networks(self):
        for t, s in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
        for t, s in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def save_model(self, filepath):
        """Save the entire model state"""
        torch.save({
            'tft_encoder_state_dict': self.tft_encoder.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'forecast_head_state_dict': self.forecast_head.state_dict(),  # Added forecast head
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'forecast_optimizer_state_dict': self.forecast_optimizer.state_dict(),  # Added forecast optimizer
        }, filepath)

    def load_model(self, filepath):
        """Load the entire model state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.tft_encoder.load_state_dict(checkpoint['tft_encoder_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.forecast_head.load_state_dict(checkpoint['forecast_head_state_dict'])  # Added forecast head
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.forecast_optimizer.load_state_dict(checkpoint['forecast_optimizer_state_dict'])  # Added forecast optimizer
        self.alpha = self.log_alpha.exp().item()