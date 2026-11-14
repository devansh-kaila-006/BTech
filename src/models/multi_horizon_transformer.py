import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHorizonTransformer(nn.Module):
    """
    Time-Series Transformer predicting across multi-step lookahead horizons:
      - Horizon 1: t+1 (Immediate pod arrival burst)
      - Horizon 2: t+5 (Transient buffer)
      - Horizon 3: t+15 (Long-range 60s physical cloud VM spin-up window)
      
    Output shape: (batch_size, 3 horizons, 2 features [cpu, mem])
    """
    def __init__(self, input_size=3, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, horizons=[1, 5, 15]):
        super(MultiHorizonTransformer, self).__init__()
        self.horizons = horizons
        self.num_horizons = len(horizons)
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Multi-Horizon Projection Heads
        # Each head predicts [cpu_demand, mem_demand] for its respective horizon
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            ) for _ in range(self.num_horizons)
        ])

    def forward(self, src):
        # src: (batch_size, seq_len, input_size)
        x = self.input_projection(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        encoded = self.transformer_encoder(x)
        last_hidden = encoded[:, -1, :] # (batch_size, d_model)
        
        # Predict across all horizons: list of (batch_size, 2)
        horizon_preds = [head(last_hidden) for head in self.horizon_heads]
        # Stack into (batch_size, num_horizons, 2)
        out = torch.stack(horizon_preds, dim=1)
        return out

