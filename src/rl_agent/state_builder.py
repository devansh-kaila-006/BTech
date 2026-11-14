import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from oracles.workload_predictor import LSTMOracle
from oracles.latency_predictor import XGBoostOracle
from oracles.power_predictor import DNNPowerOracle

import torch.nn as nn

class AttentionFusionLayer(nn.Module):
    def __init__(self, d_model=16, nhead=4):
        super(AttentionFusionLayer, self).__init__()
        # We have 4 separate "tokens": Base, Workload, Latency, Power
        self.proj_base = nn.Linear(3, d_model)
        self.proj_work = nn.Linear(2, d_model)
        self.proj_lat = nn.Linear(1, d_model)
        self.proj_pow = nn.Linear(1, d_model)
        
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.flatten = nn.Flatten()
        
    def forward(self, base, work, lat, power):
        # Project all components to the same dimension
        t_base = self.proj_base(base).unsqueeze(1)
        t_work = self.proj_work(work).unsqueeze(1)
        t_lat = self.proj_lat(lat).unsqueeze(1)
        t_pow = self.proj_pow(power).unsqueeze(1)
        
        # Stack into sequence of length 4
        # Shape: (batch, seq_len=4, d_model)
        seq = torch.cat([t_base, t_work, t_lat, t_pow], dim=1)
        
        # Self-Attention
        attn_output, attn_weights = self.attention(seq, seq, seq)
        
        # Flatten the output: (batch, 4 * d_model)
        fused_state = self.flatten(attn_output)
        return fused_state, attn_weights

class CompositeStateBuilder:
    """
    Fuses raw cluster metrics with proactive predictions from the Oracles.
    Generates the final state tensor for the PPO RL Agent.
    """
    def __init__(self, model_dir, use_forecasts=True, use_attention=False):
        self.use_forecasts = use_forecasts
        self.use_attention = use_attention
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing CompositeStateBuilder (use_forecasts={use_forecasts}, use_attention={use_attention})...")
        
        # 1. Load LSTM Workload Predictor
        self.lstm = LSTMOracle(input_dim=3, hidden_dim=32, num_layers=3, output_dim=2, dropout=0.0)
        lstm_path = os.path.join(model_dir, "best_lstm_workload.pt")
        if os.path.exists(lstm_path):
            self.lstm.load_state_dict(torch.load(lstm_path, map_location=self.device, weights_only=True))
        self.lstm.to(self.device)
        self.lstm.eval()
        
        # 2. Load XGBoost Latency Predictor
        self.xgb = XGBoostOracle()
        xgb_path = os.path.join(model_dir, "best_xgboost_latency.json")
        if os.path.exists(xgb_path):
            self.xgb.load(xgb_path)
            
        # 3. Load DNN Power Predictor
        self.dnn = DNNPowerOracle(input_dim=3, hidden_dims=[32, 32, 32], output_dim=1, dropout=0.0)
        dnn_path = os.path.join(model_dir, "best_dnn_power.pt")
        if os.path.exists(dnn_path):
            self.dnn.load_state_dict(torch.load(dnn_path, map_location=self.device, weights_only=True))
        self.dnn.to(self.device)
        self.dnn.eval()
        
        if self.use_attention:
            self.attention_fusion = AttentionFusionLayer(d_model=16, nhead=4).to(self.device)
            self.attention_fusion.eval()
            self.last_attention_weights = None
        
    def build_state(self, raw_pod_features, rolling_window_tensor):
        """
        :param raw_pod_features: dict containing current pod and node stats
        :param rolling_window_tensor: numpy array of shape (window_size, features)
        :return: 1D numpy array representing the RL state
        """
        # --- 1. Raw State Extraction ---
        base_state = np.array([
            raw_pod_features.get('cpu_norm', 0.0),
            raw_pod_features.get('mem_norm', 0.0),
            raw_pod_features.get('workload_type', 0.0)
        ], dtype=np.float32)
        
        # --- 2. Oracle Predictions ---
        if self.use_forecasts:
            with torch.no_grad():
                seq_tensor = torch.tensor(rolling_window_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
                workload_pred = self.lstm(seq_tensor).cpu().numpy().flatten()
                
                xgb_input = base_state.reshape(1, -1)
                latency_pred = self.xgb.predict(xgb_input).flatten()
                
                dnn_input = torch.tensor(base_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                power_pred = self.dnn(dnn_input).cpu().numpy().flatten()
                
            if self.use_attention:
                # Attention Fusion
                b = torch.tensor(base_state).unsqueeze(0).to(self.device)
                w = torch.tensor(workload_pred).unsqueeze(0).to(self.device)
                l = torch.tensor(latency_pred).unsqueeze(0).to(self.device)
                p = torch.tensor(power_pred).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    fused_state, attn_weights = self.attention_fusion(b, w, l, p)
                    self.last_attention_weights = attn_weights.cpu().numpy()
                return fused_state.cpu().numpy().flatten()
            else:
                # Flat Concat Fusion
                composite_state = np.concatenate([
                    base_state, workload_pred, latency_pred, power_pred
                ])
                return composite_state
        else:
            # Baseline Mode
            dummy_workload = np.zeros(2, dtype=np.float32)
            dummy_latency = np.zeros(1, dtype=np.float32)
            dummy_power = np.zeros(1, dtype=np.float32)
            composite_state = np.concatenate([
                base_state, dummy_workload, dummy_latency, dummy_power
            ])
            return composite_state

if __name__ == "__main__":
    # Quick Test
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_dir = os.path.join(PROJECT_ROOT, "models")
    builder = CompositeStateBuilder(model_dir, use_forecasts=True)
    
    mock_raw = {'cpu_norm': 0.5, 'mem_norm': 0.3, 'workload_type': 1.0}
    mock_window = np.random.rand(64, 3)
    
    state = builder.build_state(mock_raw, mock_window)
    print(f"Fused Composite State Shape: {state.shape}")
    print(f"Fused Composite State Values: {state}")
