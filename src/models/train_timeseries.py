"""
src/models/train_timeseries.py
-------------------------------
Deep learning (LSTM) model training for the BRFSS Obesity Early Warning project.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any, Tuple, Optional
from src.utils.helpers import get_logger, load_config, print_section
from src.utils.paths import model_path

logger = get_logger(__name__)

class ObesityDataset(Dataset):
    """Custom Dataset for sequence data."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMRegressor(nn.Module):
    """Simple LSTM model for regression."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int = 1, dropout: float = 0.2):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the last time step's output
        out = self.fc(out[:, -1, :])
        return out

def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_length: int = 3,
    return_years: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Prepare sequences from the dataframe.
    Assumes df is sorted by time and grouped by (LocationDesc, stratum_category, stratum_value).
    """
    sequences = []
    targets = []
    years = []
    
    group_cols = ["LocationDesc", "stratum_category", "stratum_value"]
    for _, group in df.groupby(group_cols):
        group = group.sort_values("YearStart")
        if len(group) <= seq_length:
            continue
            
        feat_data = group[feature_cols].values.astype(float)
        target_data = group[target_col].values.astype(float)
        year_data = group["YearStart"].values
        
        for i in range(len(group) - seq_length):
            sequences.append(feat_data[i:i + seq_length])
            targets.append(target_data[i + seq_length])
            years.append(year_data[i + seq_length])
            
    X = np.array(sequences, dtype=float)
    y = np.array(targets, dtype=float)
    years_arr = np.array(years)
    
    if return_years:
        return X, y, years_arr
    return X, y

def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any]
) -> nn.Module:
    """Train the LSTM model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Hyperparameters
    input_size = X_train.shape[2]
    hidden_size = config.get("timeseries", {}).get("hidden_size", 64)
    num_layers = config.get("timeseries", {}).get("num_layers", 2)
    learning_rate = config.get("timeseries", {}).get("learning_rate", 0.001)
    batch_size = config.get("timeseries", {}).get("batch_size", 32)
    num_epochs = config.get("timeseries", {}).get("num_epochs", 50)
    
    model = LSTMRegressor(input_size, hidden_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = ObesityDataset(X_train, y_train)
    val_dataset = ObesityDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save best model
            save_path = model_path("timeseries", "lstm_best.pth")
            torch.save(model.state_dict(), save_path)
            
    logger.info(f"LSTM training complete. Best Val Loss: {best_val_loss:.4f}")
    return model
