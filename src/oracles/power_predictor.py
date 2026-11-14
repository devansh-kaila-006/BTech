import torch
import torch.nn as nn

class DNNPowerOracle(nn.Module):
    """
    Primary Power Predictor: Deep Neural Network.
    Maps non-linear node utilization features (CPU/GPU saturation, Memory overhead)
    to a continuous power draw estimate (Watts).
    """
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.1):
        """
        :param input_dim: Size of the node utilization feature vector.
        :param hidden_dims: List of integers representing hidden layer sizes.
        """
        super(DNNPowerOracle, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        return self.network(x)
