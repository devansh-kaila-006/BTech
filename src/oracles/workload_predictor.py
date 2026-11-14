import torch
import torch.nn as nn

class LSTMOracle(nn.Module):
    """
    Primary Workload Predictor: Long Short-Term Memory Network.
    Designed to process the sequential output of the RollingWindowBuffer
    and forecast the aggregate CPU/Memory burst for the next timestep.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMOracle, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class TransformerOracle(nn.Module):
    """
    Ablation Workload Predictor: Attention-based Sequence Model.
    Tests if self-attention captures long-term burst signatures better than LSTM.
    """
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim, dropout=0.2, seq_len=64):
        super(TransformerOracle, self).__init__()
        
        # Linear projection to model dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Project and add positional encoding
        x = self.input_projection(x) + self.positional_encoding[:, :x.size(1), :]
        
        # Pass through Transformer
        out = self.transformer_encoder(x)
        
        # Pool the sequence (mean pooling)
        out = out.mean(dim=1)
        
        # Regression head
        out = self.fc(out)
        return out
