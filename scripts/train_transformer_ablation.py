import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.models.transformer_predictor import WorkloadTransformer
from src.oracles.workload_predictor import LSTMOracle # The original LSTM

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_trace.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ABLATION_DIR = os.path.join(PROJECT_ROOT, "ablations")

def prepare_sequence_data(df, window_size=64, subset_size=20000):
    df_sub = df.head(subset_size).copy()
    
    # We predict the next timestamp's cpu and mem based on the window
    features = ['cpu_norm', 'mem_norm', 'workload_type']
    X_data = df_sub[features].values
    y_data = df_sub[['cpu_norm', 'mem_norm']].shift(-1).fillna(method='ffill').values
    
    X, y = [], []
    for i in range(len(X_data) - window_size):
        X.append(X_data[i:i+window_size])
        y.append(y_data[i+window_size-1])
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y

def train_transformer(X_train, y_train, epochs=10):
    print("Training Transformer Oracle...")
    model = WorkloadTransformer(input_size=3, output_size=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model.train()
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Transformer Loss: {avg_loss:.6f}")
        
    # Save the model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "transformer_workload.pth"))
    return model, loss_history

if __name__ == "__main__":
    print("Loading data for Transformer Ablation...")
    df = pd.read_parquet(DATA_PATH)
    
    # Use 20k rows for train, next 5k for test
    X_train, y_train = prepare_sequence_data(df, subset_size=20000)
    
    # Test set (starting from row 20000)
    df_test = df.iloc[20000:25000].reset_index(drop=True)
    X_test, y_test = prepare_sequence_data(df_test, subset_size=5000)
    
    # Train Transformer
    transformer_model, trans_losses = train_transformer(X_train, y_train, epochs=15)
    
    # Load LSTM
    print("Loading Baseline LSTM...")
    # Initialize with the exact trained architecture from Stage 2 (hidden_dim=32, num_layers=3)
    lstm_model = LSTMOracle(input_dim=3, hidden_dim=32, num_layers=3, output_dim=2, dropout=0.0) 
    lstm_path = os.path.join(MODEL_DIR, "best_lstm_workload.pt")
    if os.path.exists(lstm_path):
        lstm_model.load_state_dict(torch.load(lstm_path, map_location='cpu', weights_only=True))
        print("Successfully loaded pre-trained LSTM weights.")
    else:
        print("Warning: pre-trained LSTM not found. Initializing random LSTM.")
        
    # Evaluate
    print("\nEvaluating on unseen test set...")
    transformer_model.eval()
    lstm_model.eval()
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        trans_preds = transformer_model(X_test)
        trans_test_loss = criterion(trans_preds, y_test).item()
        
        lstm_preds = lstm_model(X_test)
        lstm_test_loss = criterion(lstm_preds, y_test).item()
        
    print(f"Final LSTM Test MSE:        {lstm_test_loss:.6f}")
    print(f"Final Transformer Test MSE: {trans_test_loss:.6f}")
    
    # Generate Plots
    print("Generating Ablation Graphs...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    # 1. Loss Comparison
    axes[0].plot(trans_losses, label="Transformer Train Loss", color='blue')
    axes[0].set_title("Transformer Training Convergence")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. CPU Prediction Plot
    y_test_cpu = y_test[:300, 0].numpy()
    lstm_cpu = lstm_preds[:300, 0].numpy()
    trans_cpu = trans_preds[:300, 0].numpy()
    
    axes[1].plot(y_test_cpu, label="Actual CPU Workload", color='black', alpha=0.7)
    axes[1].plot(lstm_cpu, label=f"LSTM (MSE: {lstm_test_loss:.4f})", color='red', linestyle='--')
    axes[1].plot(trans_cpu, label=f"Transformer (MSE: {trans_test_loss:.4f})", color='green')
    axes[1].set_title("CPU Workload Prediction (Test Set, 300 Steps)")
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. Memory Prediction Plot
    y_test_mem = y_test[:300, 1].numpy()
    lstm_mem = lstm_preds[:300, 1].numpy()
    trans_mem = trans_preds[:300, 1].numpy()
    
    axes[2].plot(y_test_mem, label="Actual Mem Workload", color='black', alpha=0.7)
    axes[2].plot(lstm_mem, label="LSTM", color='red', linestyle='--')
    axes[2].plot(trans_mem, label="Transformer", color='green')
    axes[2].set_title("Memory Workload Prediction (Test Set, 300 Steps)")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(ABLATION_DIR, "stage7_transformer_vs_lstm.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")
    
    # Write report
    report_path = os.path.join(ABLATION_DIR, "stage7_transformer_ablation.md")
    with open(report_path, "w") as f:
        f.write("# Stage 7 Ablation: Transformer vs LSTM\n\n")
        f.write(f"We replaced the traditional LSTM Workload Predictor with a PyTorch Time-Series Transformer.\n\n")
        f.write(f"## Test Set Metrics (MSE)\n")
        f.write(f"- **LSTM MSE:** `{lstm_test_loss:.6f}`\n")
        f.write(f"- **Transformer MSE:** `{trans_test_loss:.6f}`\n\n")
        f.write(f"## Analysis\n")
        f.write(f"*(Include analysis here based on the graphs)*\n\n")
        f.write(f"![Transformer vs LSTM Plot](file:///{save_path.replace(chr(92), '/')})\n")
    print(f"Report written to {report_path}")
