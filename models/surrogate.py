"""
Surrogate models: predict design-point performance (cd, alpha, cm at the
design cl) from the 12 CST weights, so the optimiser can skip the solver.

Model types: 'ridge' (linear baseline), 'rf', 'xgboost', 'nn' (PyTorch MLP)
and 'ensemble' (mean of rf, xgboost and nn). Each row of the data is one
shape, so a split by row is a split by shape.
"""

from typing import Dict

import joblib
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

TARGETS = ['cd', 'alpha', 'cm']
N_JOBS = 2
torch.set_num_threads(N_JOBS)


class AirfoilNeuralNetwork(nn.Module):
    """Fully connected network, CST weights -> scaled targets."""

    def __init__(self, input_dim: int = 12, hidden_dims=(64, 64), output_dim: int = 3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SurrogateModel:
    def __init__(self, model_type: str = 'nn', seed: int = 0):
        self.model_type = model_type
        self.seed = seed
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.models: Dict[str, object] = {}
        self.history: Dict[str, list] = {}

    def _members(self):
        return ['rf', 'xgboost', 'nn'] if self.model_type == 'ensemble' else [self.model_type]

    def fit(self, X_train, y_train, X_val, y_val, epochs: int = 3000, patience: int = 200):
        """Fit on the training set. The validation set is used only for
        early stopping of the network, never for fitting weights."""
        Xt = self.scaler_X.fit_transform(X_train)
        Xv = self.scaler_X.transform(X_val)
        yt = self.scaler_y.fit_transform(y_train)
        yv = self.scaler_y.transform(y_val)
        for name in self._members():
            if name == 'ridge':
                self.models[name] = Ridge(alpha=1e-3).fit(Xt, yt)
            elif name == 'rf':
                self.models[name] = RandomForestRegressor(
                    n_estimators=300, min_samples_leaf=2, random_state=self.seed,
                    n_jobs=N_JOBS).fit(Xt, yt)
            elif name == 'xgboost':
                self.models[name] = [
                    xgb.XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=self.seed, n_jobs=N_JOBS).fit(Xt, yt[:, i])
                    for i in range(yt.shape[1])]
            elif name == 'nn':
                self.models[name] = self._fit_nn(Xt, yt, Xv, yv, epochs, patience)
            else:
                raise ValueError(f"Unknown model type: {name}")
        return self

    def _fit_nn(self, Xt, yt, Xv, yv, epochs, patience, batch_size=64, lr=2e-3):
        torch.manual_seed(self.seed)
        net = AirfoilNeuralNetwork(Xt.shape[1], output_dim=yt.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=50)
        Xt_t, yt_t = torch.tensor(Xt, dtype=torch.float32), torch.tensor(yt, dtype=torch.float32)
        Xv_t, yv_t = torch.tensor(Xv, dtype=torch.float32), torch.tensor(yv, dtype=torch.float32)
        loss_fn = nn.MSELoss()
        best, best_state, wait = np.inf, None, 0
        self.history = {'train': [], 'val': []}
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(len(Xt_t))
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Xt_t[idx]), yt_t[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                train_loss = loss_fn(net(Xt_t), yt_t).item()
                val_loss = loss_fn(net(Xv_t), yv_t).item()
            self.history['train'].append(train_loss)
            self.history['val'].append(val_loss)
            sched.step(val_loss)
            if val_loss < best - 1e-7:
                best, wait = val_loss, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return net

    def _predict_member(self, name, Xs):
        m = self.models[name]
        if name == 'nn':
            with torch.no_grad():
                return m(torch.tensor(Xs, dtype=torch.float32)).numpy()
        if name == 'xgboost':
            return np.column_stack([mi.predict(Xs) for mi in m])
        return m.predict(Xs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_X.transform(np.atleast_2d(X))
        pred = np.mean([self._predict_member(n, Xs) for n in self._members()], axis=0)
        return self.scaler_y.inverse_transform(pred)

    def save(self, path: str):
        state = dict(self.__dict__)
        if 'nn' in self.models:
            state['models'] = dict(self.models)
            state['models']['nn'] = self.models['nn'].state_dict()
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "SurrogateModel":
        state = joblib.load(path)
        obj = cls(state['model_type'], state['seed'])
        obj.__dict__.update(state)
        if 'nn' in obj.models:
            net = AirfoilNeuralNetwork(obj.scaler_X.n_features_in_,
                                       output_dim=obj.scaler_y.n_features_in_)
            net.load_state_dict(obj.models['nn'])
            net.eval()
            obj.models = dict(obj.models, nn=net)
        return obj


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """R^2 and mean absolute error for each target, plus cd error in percent."""
    out = {}
    for i, name in enumerate(TARGETS):
        out[name] = {'r2': float(r2_score(y_true[:, i], y_pred[:, i])),
                     'mae': float(mean_absolute_error(y_true[:, i], y_pred[:, i]))}
    rel = np.abs(y_pred[:, 0] / y_true[:, 0] - 1)
    out['cd']['mean_abs_pct'] = float(100 * rel.mean())
    out['cd']['p95_abs_pct'] = float(100 * np.percentile(rel, 95))
    return out
