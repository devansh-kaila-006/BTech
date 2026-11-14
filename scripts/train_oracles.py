import os
import sys
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.oracles.workload_predictor import LSTMOracle, TransformerOracle
from src.oracles.latency_predictor import XGBoostOracle
from src.oracles.power_predictor import DNNPowerOracle

# Configuration
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_trace.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class OracleTrainer:
    def __init__(self, data_path):
        print("Loading data for Oracle Training...")
        self.df = pd.read_parquet(data_path)
        # Sort chronologically to prevent temporal data leakage in sequence models
        self.df = self.df.sort_values(by='creation_time').reset_index(drop=True)
        
    def _create_sequence_data(self, window_size=64):
        """Creates sliding windows for the Workload Predictor."""
        # For demonstration/testing, we use a subset to keep training fast
        df_sub = self.df.head(10000)
        features = df_sub[['cpu_norm', 'mem_norm', 'workload_type']].values
        targets = df_sub[['cpu_norm', 'mem_norm']].values # predicting the next pod's demand
        
        X, y = [], []
        for i in range(len(features) - window_size):
            X.append(features[i:i+window_size])
            y.append(targets[i+window_size])
            
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)
        return torch.utils.data.TensorDataset(X, y)

    def optimize_workload_predictor(self, n_trials=10):
        print("\n--- Running HPO for LSTM Workload Predictor ---")
        dataset = self._create_sequence_data()
        train_size = int(0.8 * len(dataset))
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=256)
        
        def objective(trial):
            # Hyperparameter search space
            hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
            num_layers = trial.suggest_int('num_layers', 1, 3)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            
            model = LSTMOracle(input_dim=3, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=2, dropout=dropout)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            # Fast training loop (1 epoch for HPO demo)
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    out = model(batch_X)
                    val_loss += criterion(out, batch_y).item()
                    
            return val_loss / len(val_loader)
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        print(f"Best LSTM Params: {study.best_params}")
        
        # Train and save the absolute best model
        best_model = LSTMOracle(
            input_dim=3, 
            hidden_dim=study.best_params['hidden_dim'], 
            num_layers=study.best_params['num_layers'], 
            output_dim=2, 
            dropout=study.best_params['dropout']
        )
        optimizer = optim.Adam(best_model.parameters(), lr=study.best_params['lr'])
        criterion = nn.MSELoss()
        
        best_model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = best_model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
        torch.save(best_model.state_dict(), os.path.join(MODEL_DIR, "best_lstm_workload.pt"))
        print("Saved best_lstm_workload.pt")
        
    def optimize_latency_predictor(self, n_trials=10):
        print("\n--- Running HPO for XGBoost Latency Predictor ---")
        df_sub = self.df.dropna(subset=['scheduling_delay']).head(20000)
        X = df_sub[['cpu_norm', 'mem_norm', 'workload_type']].values
        y = df_sub['scheduling_delay'].values
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
            
            model = XGBoostOracle(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return mean_squared_error(y_val, preds)
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        print(f"Best XGBoost Params: {study.best_params}")
        
        # Save best model
        best_model = XGBoostOracle(**study.best_params)
        best_model.fit(X_train, y_train)
        best_model.save(os.path.join(MODEL_DIR, "best_xgboost_latency.json"))

    def optimize_power_predictor(self, n_trials=10):
        print("\n--- Running HPO for DNN Power Predictor ---")
        # Synthesize power label based on standard cloud TDP models since the raw trace lacks Watts
        # Power = Idle (100W) + Max_Draw (250W) * CPU_Utilization
        df_sub = self.df.head(20000).copy()
        df_sub['power_watts'] = 100.0 + (250.0 * df_sub['cpu_norm']) + (df_sub['workload_type'] * 150.0) # Add GPU overhead
        
        X = torch.tensor(df_sub[['cpu_norm', 'mem_norm', 'workload_type']].values, dtype=torch.float32)
        y = torch.tensor(df_sub[['power_watts']].values, dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=256)
        
        def objective(trial):
            layer_size = trial.suggest_categorical('layer_size', [32, 64, 128])
            depth = trial.suggest_int('depth', 1, 3)
            hidden_dims = [layer_size] * depth
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float('dropout', 0.05, 0.3)
            
            model = DNNPowerOracle(input_dim=3, hidden_dims=hidden_dims, output_dim=1, dropout=dropout)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    out = model(batch_X)
                    val_loss += criterion(out, batch_y).item()
                    
            return val_loss / len(val_loader)
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        print(f"Best DNN Power Params: {study.best_params}")

        # Train and save the absolute best model
        hidden_dims = [study.best_params['layer_size']] * study.best_params['depth']
        best_model = DNNPowerOracle(
            input_dim=3, 
            hidden_dims=hidden_dims, 
            output_dim=1, 
            dropout=study.best_params['dropout']
        )
        optimizer = optim.Adam(best_model.parameters(), lr=study.best_params['lr'])
        criterion = nn.MSELoss()
        
        best_model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = best_model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
        torch.save(best_model.state_dict(), os.path.join(MODEL_DIR, "best_dnn_power.pt"))
        print("Saved best_dnn_power.pt")

if __name__ == "__main__":
    trainer = OracleTrainer(DATA_PATH)
    # We run small trials to prove the architecture works
    trainer.optimize_workload_predictor(n_trials=3)
    trainer.optimize_latency_predictor(n_trials=5)
    trainer.optimize_power_predictor(n_trials=3)
    print("\nAll Oracle HPO sweeps completed and best models saved!")
