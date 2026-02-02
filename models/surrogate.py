import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib
from typing import Dict, Tuple, Optional
import json


class AirfoilNeuralNetwork(nn.Module):
    """
    Deep neural network for airfoil performance prediction
    """
    
    def __init__(self, input_dim: int = 11, hidden_dims: list = [128, 256, 256, 128],
                 output_dim: int = 4, dropout: float = 0.2):
        """
        Initialize neural network
        
        Args:
            input_dim: Number of input features (PARSEC parameters + alpha)
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of outputs (cl, cd, cm, ld_ratio)
            dropout: Dropout probability
        """
        super(AirfoilNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class SurrogateModel:
    """
    Ensemble surrogate model for airfoil performance prediction
    Combines neural network, XGBoost, and random forest
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize surrogate model
        
        Args:
            model_type: 'nn', 'xgboost', 'rf', or 'ensemble'
        """
        self.model_type = model_type
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Initialize models
        self.nn_model = None
        self.xgb_model = None
        self.rf_model = None
        
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2,
              epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.001) -> Dict:
        """
        Train the surrogate model
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target outputs (n_samples, n_targets)
            validation_split: Fraction of data for validation
            epochs: Number of training epochs for NN
            batch_size: Batch size for NN training
            learning_rate: Learning rate for NN
            
        Returns:
            Dictionary with training metrics
        """
        print(f"Training {self.model_type} surrogate model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Scale data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        
        metrics = {}
        
        # Train Neural Network
        if self.model_type in ['nn', 'ensemble']:
            print("  Training Neural Network...")
            nn_metrics = self._train_nn(
                X_train_scaled, y_train_scaled,
                X_val_scaled, y_val_scaled,
                epochs, batch_size, learning_rate
            )
            metrics['nn'] = nn_metrics
        
        # Train XGBoost
        if self.model_type in ['xgboost', 'ensemble']:
            print("  Training XGBoost...")
            xgb_metrics = self._train_xgboost(
                X_train_scaled, y_train_scaled,
                X_val_scaled, y_val_scaled
            )
            metrics['xgboost'] = xgb_metrics
        
        # Train Random Forest
        if self.model_type in ['rf', 'ensemble']:
            print("  Training Random Forest...")
            rf_metrics = self._train_rf(
                X_train_scaled, y_train_scaled,
                X_val_scaled, y_val_scaled
            )
            metrics['rf'] = rf_metrics
        
        self.is_trained = True
        print("Training complete!")
        
        return metrics
    
    def _train_nn(self, X_train, y_train, X_val, y_val,
                  epochs, batch_size, learning_rate) -> Dict:
        """Train neural network"""
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        
        self.nn_model = AirfoilNeuralNetwork(input_dim=input_dim, output_dim=output_dim)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.nn_model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.nn_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.nn_model.eval()
            with torch.no_grad():
                val_outputs = self.nn_model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Calculate R² score
        with torch.no_grad():
            y_pred = self.nn_model(X_val_t).numpy()
        r2 = r2_score(y_val, y_pred, multioutput='uniform_average')
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'r2_score': r2
        }
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train XGBoost models (one per output)"""
        n_outputs = y_train.shape[1]
        self.xgb_model = []
        
        r2_scores = []
        
        for i in range(n_outputs):
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train, y_train[:, i],
                     eval_set=[(X_val, y_val[:, i])],
                     verbose=False)
            
            self.xgb_model.append(model)
            
            # Calculate R² score
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val[:, i], y_pred)
            r2_scores.append(r2)
        
        return {
            'r2_scores': r2_scores,
            'avg_r2': np.mean(r2_scores)
        }
    
    def _train_rf(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train Random Forest models (one per output)"""
        n_outputs = y_train.shape[1]
        self.rf_model = []
        
        r2_scores = []
        
        for i in range(n_outputs):
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train[:, i])
            
            self.rf_model.append(model)
            
            # Calculate R² score
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val[:, i], y_pred)
            r2_scores.append(r2)
        
        return {
            'r2_scores': r2_scores,
            'avg_r2': np.mean(r2_scores)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict performance for given inputs
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted outputs (n_samples, n_targets)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler_X.transform(X)
        
        predictions = []
        
        # Get predictions from each model
        if self.nn_model is not None:
            self.nn_model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_scaled)
                nn_pred = self.nn_model(X_t).numpy()
            predictions.append(nn_pred)
        
        if self.xgb_model is not None:
            xgb_pred = np.column_stack([model.predict(X_scaled) for model in self.xgb_model])
            predictions.append(xgb_pred)
        
        if self.rf_model is not None:
            rf_pred = np.column_stack([model.predict(X_scaled) for model in self.rf_model])
            predictions.append(rf_pred)
        
        # Ensemble: average predictions
        if self.model_type == 'ensemble' and len(predictions) > 1:
            y_pred_scaled = np.mean(predictions, axis=0)
        else:
            y_pred_scaled = predictions[0]
        
        # Inverse transform
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def save(self, filepath: str):
        """Save model to disk"""
        save_dict = {
            'model_type': self.model_type,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'is_trained': self.is_trained
        }
        
        # Save neural network separately
        if self.nn_model is not None:
            torch.save(self.nn_model.state_dict(), filepath.replace('.pkl', '_nn.pth'))
        
        # Save other models
        if self.xgb_model is not None:
            save_dict['xgb_model'] = self.xgb_model
        if self.rf_model is not None:
            save_dict['rf_model'] = self.rf_model
        
        joblib.dump(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        save_dict = joblib.load(filepath)
        
        self.model_type = save_dict['model_type']
        self.scaler_X = save_dict['scaler_X']
        self.scaler_y = save_dict['scaler_y']
        self.is_trained = save_dict['is_trained']
        
        # Load neural network
        if 'nn' in self.model_type or self.model_type == 'ensemble':
            input_dim = self.scaler_X.n_features_in_
            output_dim = self.scaler_y.n_features_in_
            self.nn_model = AirfoilNeuralNetwork(input_dim=input_dim, output_dim=output_dim)
            self.nn_model.load_state_dict(torch.load(filepath.replace('.pkl', '_nn.pth')))
            self.nn_model.eval()
        
        # Load other models
        if 'xgb_model' in save_dict:
            self.xgb_model = save_dict['xgb_model']
        if 'rf_model' in save_dict:
            self.rf_model = save_dict['rf_model']
        
        print(f"Model loaded from {filepath}")


def prepare_training_data(database_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from simulation database
    
    Args:
        database_file: Path to .npz database file
        
    Returns:
        X (features), y (targets) arrays
    """
    data = np.load(database_file)
    
    params = data['params']  # (n_samples, 11)
    cl = data['cl']          # (n_samples, n_alphas)
    cd = data['cd']
    cm = data['cm']
    ld = data['ld_ratio']
    alphas = data['alphas']
    
    n_samples, n_params = params.shape
    n_alphas = len(alphas)
    
    # Create feature matrix: [PARSEC params, alpha]
    X = []
    y = []
    
    for i in range(n_samples):
        for j in range(n_alphas):
            # Features: 11 PARSEC params + 1 alpha
            features = np.concatenate([params[i], [alphas[j]]])
            X.append(features)
            
            # Targets: [cl, cd, cm, ld_ratio]
            targets = [cl[i, j], cd[i, j], cm[i, j], ld[i, j]]
            y.append(targets)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
    
    return X, y


if __name__ == "__main__":
    print("Machine Learning Surrogate Model Module")
    print("=" * 50)
    
    # Example: Create small dataset and train model
    from simulation.aerodynamics import create_performance_database
    
    # Generate small database
    db_file = '/home/claude/airfoil_optimization/data/small_database.npz'
    create_performance_database(n_samples=200, output_file=db_file)
    
    # Prepare training data
    X, y = prepare_training_data(db_file)
    
    # Train surrogate model
    model = SurrogateModel(model_type='ensemble')
    metrics = model.train(X, y, epochs=50, batch_size=32)
    
    print("\nTraining Metrics:")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name.upper()}:")
        if 'r2_score' in model_metrics:
            print(f"  R² Score: {model_metrics['r2_score']:.4f}")
        if 'avg_r2' in model_metrics:
            print(f"  Average R²: {model_metrics['avg_r2']:.4f}")
    
    # Save model
    model.save('/home/claude/airfoil_optimization/models/surrogate_model.pkl')
    
    # Test prediction
    print("\nTesting prediction...")
    test_sample = X[:5]  # First 5 samples
    predictions = model.predict(test_sample)
    
    print("\nPredictions vs Actual:")
    print("Sample | CL_pred | CL_true | CD_pred | CD_true")
    for i in range(5):
        print(f"  {i+1}    | {predictions[i,0]:7.4f} | {y[i,0]:7.4f} | {predictions[i,1]:7.4f} | {y[i,1]:7.4f}")