import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class WorkloadTransformer(nn.Module):
    """
    Time-Series Transformer for forecasting future workloads.
    Provides a highly attentive alternative to the LSTM Oracle.
    """
    def __init__(self, input_size=3, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, output_size=2):
        super(WorkloadTransformer, self).__init__()
        self.model_type = 'Transformer'
        
        # Linear layer to project raw input features to d_model dimension
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.d_model = d_model
        
        # Final output projection
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_size)
        x = self.input_projection(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Pass through Transformer Encoder
        output = self.transformer_encoder(x)
        
        # We only care about predicting based on the last time step in the sequence
        # output shape: (batch_size, seq_len, d_model)
        last_step = output[:, -1, :] 
        
        pred = self.decoder(last_step)
        return pred
