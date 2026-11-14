import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

class XGBoostOracle:
    """
    Primary Latency Predictor: Extreme Gradient Boosting.
    Predicts the `scheduling_delay` based on incoming pod features
    and current node capacity vector. Highly efficient for tabular data.
    """
    def __init__(self, **params):
        # Default hyperparams, to be overwritten by Optuna
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1
        }
        default_params.update(params)
        self.model = xgb.XGBRegressor(**default_params)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def save(self, path):
        self.model.save_model(path)
        
    def load(self, path):
        self.model.load_model(path)


class RandomForestOracle:
    """
    Ablation Latency Predictor: Standard Random Forest.
    Tests if gradient boosting significantly outperforms bagging
    for scheduling delay prediction.
    """
    def __init__(self, **params):
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'n_jobs': -1
        }
        default_params.update(params)
        self.model = RandomForestRegressor(**default_params)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    # Note: sklearn models are usually saved via joblib, which we will handle in the train script
