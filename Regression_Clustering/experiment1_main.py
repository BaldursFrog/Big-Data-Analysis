import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
import multiprocessing as mp

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
np.random.seed(42)
tf.random.set_seed(42)

class ExperimentConfig:
    def __init__(self):
        self.random_seeds = [42, 123, 456, 789, 999]
        self.test_sizes = [0.3, 0.2, 0.1]
        self.sample_sizes = [100, 500, 1000, 5000, 10000]
        self.feature_dims = [1, 5, 10, 20, 50]
        self.noise_levels = [0.1, 0.5, 1.0]
        
        self.nn_params = {
            'hidden_layers': [1, 2, 3],
            'neurons_per_layer': [32, 64, 128],
            'learning_rates': [0.001, 0.01, 0.1],
            'batch_sizes': [16, 32, 64],
            'dropout_rates': [0.0, 0.2, 0.5],
            'l2_reg': [0.0, 0.01, 0.1]
        }
        
        self.linear_params = {
            'alpha_ridge': [0.01, 0.1, 1.0, 10.0],
            'alpha_lasso': [0.01, 0.1, 1.0, 10.0],
            'max_iter': [1000, 5000],
            'tolerance': [1e-4, 1e-3]
        }

class DataGenerator:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def generate_linear_data(self, n_samples: int, n_features: int, noise_level: float, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if random_state:
            np.random.seed(random_state)
            
        X = np.random.randn(n_samples, n_features)
        true_weights = np.random.randn(n_features)
        true_bias = np.random.randn()
        
        y = X @ true_weights + true_bias + noise_level * np.random.randn(n_samples)
        
        return X, y
    
    def generate_polynomial_data(self, n_samples: int, n_features: int, noise_level: float, degree: int = 2, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if random_state:
            np.random.seed(random_state)
            
        X = np.random.randn(n_samples, n_features)
        true_weights = np.random.randn(n_features)
        true_bias = np.random.randn()
        
        linear_term = X @ true_weights
        poly_term = np.sum(X[:, :min(n_features, degree)]**2, axis=1)
        y = linear_term + 0.5 * poly_term + true_bias + noise_level * np.random.randn(n_samples)
        
        return X, y
    
    def generate_exponential_data(self, n_samples: int, n_features: int, noise_level: float, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if random_state:
            np.random.seed(random_state)
            
        X = np.random.randn(n_samples, n_features)
        true_weights = np.random.randn(n_features)
        true_bias = np.random.randn()
        
        linear_term = X @ true_weights
        exp_term = 0.1 * np.exp(linear_term)
        y = exp_term + true_bias + noise_level * np.random.randn(n_samples)
        
        return X, y
    
    def generate_interaction_data(self, n_samples: int, n_features: int, noise_level: float, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if random_state:
            np.random.seed(random_state)
            
        X = np.random.randn(n_samples, n_features)
        true_weights = np.random.randn(n_features)
        true_bias = np.random.randn()
        
        linear_term = X @ true_weights
        if n_features >= 2:
            interaction_term = 0.5 * X[:, 0] * X[:, 1]
        else:
            interaction_term = 0.5 * X[:, 0]**2
            
        y = linear_term + interaction_term + true_bias + noise_level * np.random.randn(n_samples)
        
        return X, y
    
    def generate_piecewise_data(self, n_samples: int, n_features: int, noise_level: float, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if random_state:
            np.random.seed(random_state)
            
        X = np.random.randn(n_samples, n_features)
        true_weights = np.random.randn(n_features)
        true_bias = np.random.randn()
        
        linear_term = X @ true_weights
        piecewise_term = np.where(linear_term > 0, 2 * linear_term, 0.5 * linear_term)
        y = piecewise_term + true_bias + noise_level * np.random.randn(n_samples)
        
        return X, y
    
    def generate_synthetic_datasets(self) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, Dict]]]:
        datasets = {}
        
        for sample_size in self.config.sample_sizes:
            for n_features in self.config.feature_dims:
                for noise_level in self.config.noise_levels:
                    dataset_key = f"n{sample_size}_f{n_features}_noise{noise_level}"
                    datasets[dataset_key] = []
                    
                    for seed in self.config.random_seeds:
                        linear_data = self.generate_linear_data(sample_size, n_features, noise_level, seed)
                        poly_data = self.generate_polynomial_data(sample_size, n_features, noise_level, random_state=seed)
                        exp_data = self.generate_exponential_data(sample_size, n_features, noise_level, seed)
                        interaction_data = self.generate_interaction_data(sample_size, n_features, noise_level, seed)
                        piecewise_data = self.generate_piecewise_data(sample_size, n_features, noise_level, seed)
                        
                        datasets[dataset_key].extend([
                            (linear_data[0], linear_data[1], {"type": "linear", "sample_size": sample_size, "n_features": n_features, "noise": noise_level}),
                            (poly_data[0], poly_data[1], {"type": "polynomial", "sample_size": sample_size, "n_features": n_features, "noise": noise_level}),
                            (exp_data[0], exp_data[1], {"type": "exponential", "sample_size": sample_size, "n_features": n_features, "noise": noise_level}),
                            (interaction_data[0], interaction_data[1], {"type": "interaction", "sample_size": sample_size, "n_features": n_features, "noise": noise_level}),
                            (piecewise_data[0], piecewise_data[1], {"type": "piecewise", "sample_size": sample_size, "n_features": n_features, "noise": noise_level})
                        ])
        
        return datasets
    
    def load_california_housing(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        metadata = {
            "type": "california_housing",
            "n_features": X.shape[1],
            "sample_size": X.shape[0],
            "feature_names": housing.feature_names.tolist(),
            "description": housing.DESCR
        }
        return X, y, metadata

class DataPreprocessor:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.scalers = {}
        
    def preprocess_synthetic_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Dict:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
        
        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train_scaled,
            "y_test": y_test_scaled,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "X_train_original": X_train,
            "X_test_original": X_test,
            "y_train_original": y_train,
            "y_test_original": y_test
        }
    
    def preprocess_california_housing(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Dict:
        return self.preprocess_synthetic_data(X, y, test_size, random_state)
    
    def create_polynomial_features(self, X: np.ndarray, degree: int = 2) -> np.ndarray:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(X)

class LinearRegressionModels:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.models = {}
        self.best_params = {}
        
    def create_ols_model(self) -> LinearRegression:
        return LinearRegression()
    
    def create_ridge_model(self, alpha: float = 1.0) -> Ridge:
        return Ridge(alpha=alpha, random_state=42)
    
    def create_lasso_model(self, alpha: float = 1.0) -> Lasso:
        return Lasso(alpha=alpha, random_state=42, max_iter=5000)
    
    def tune_ridge_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        param_grid = {'alpha': self.config.linear_params['alpha_ridge']}
        ridge = Ridge(random_state=42)
        grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        self.best_params['ridge'] = grid_search.best_params_
        return grid_search.best_params_
    
    def tune_lasso_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        param_grid = {'alpha': self.config.linear_params['alpha_lasso']}
        lasso = Lasso(random_state=42, max_iter=5000)
        grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        self.best_params['lasso'] = grid_search.best_params_
        return grid_search.best_params_
    
    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, model_type: str = 'ols') -> Dict:
        start_time = time.time()
        
        if model_type == 'ols':
            model = self.create_ols_model()
        elif model_type == 'ridge':
            best_params = self.tune_ridge_hyperparameters(X_train, y_train)
            model = self.create_ridge_model(best_params['alpha'])
        elif model_type == 'lasso':
            best_params = self.tune_lasso_hyperparameters(X_train, y_train)
            model = self.create_lasso_model(best_params['alpha'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results = {
            'model': model,
            'model_type': model_type,
            'training_time': training_time,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'coefficients': model.coef_ if hasattr(model, 'coef_') else None,
            'intercept': model.intercept_ if hasattr(model, 'intercept_') else None
        }
        
        if model_type in ['ridge', 'lasso']:
            results['best_params'] = self.best_params.get(model_type, {})
        
        return results

class NeuralNetworkModels:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.models = {}
        self.best_params = {}
        
    def create_model(self, input_dim: int, hidden_layers: int = 2, neurons_per_layer: int = 64, 
                    dropout_rate: float = 0.2, l2_reg: float = 0.01, learning_rate: float = 0.001) -> keras.Model:
        model = Sequential()
        
        model.add(layers.Dense(neurons_per_layer, input_dim=input_dim, activation='relu', 
                              kernel_regularizer=keras.regularizers.l2(l2_reg)))
        model.add(layers.Dropout(dropout_rate))
        
        for _ in range(hidden_layers - 1):
            model.add(layers.Dense(neurons_per_layer, activation='relu', 
                                  kernel_regularizer=keras.regularizers.l2(l2_reg)))
            model.add(layers.Dropout(dropout_rate))
        
        model.add(layers.Dense(1))
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, input_dim: int) -> Dict:
        best_score = float('-inf')
        best_params = {}
        
        for hidden_layers in self.config.nn_params['hidden_layers']:
            for neurons in self.config.nn_params['neurons_per_layer']:
                for lr in self.config.nn_params['learning_rates']:
                    for dropout in self.config.nn_params['dropout_rates']:
                        for l2 in self.config.nn_params['l2_reg']:
                            model = self.create_model(input_dim, hidden_layers, neurons, dropout, l2, lr)
                            
                            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                            
                            history = model.fit(X_train, y_train, 
                                              epochs=100, batch_size=32, 
                                              validation_split=0.2, 
                                              callbacks=[early_stopping], 
                                              verbose=0)
                            
                            val_score = max(history.history['val_mae'])
                            
                            if val_score > best_score:
                                best_score = val_score
                                best_params = {
                                    'hidden_layers': hidden_layers,
                                    'neurons_per_layer': neurons,
                                    'learning_rate': lr,
                                    'dropout_rate': dropout,
                                    'l2_reg': l2
                                }
        
        self.best_params['neural_network'] = best_params
        return best_params
    
    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, 
                          tune_hyperparams: bool = True) -> Dict:
        input_dim = X_train.shape[1]
        
        start_time = time.time()
        
        if tune_hyperparams:
            best_params = self.tune_hyperparameters(X_train, y_train, input_dim)
        else:
            best_params = {
                'hidden_layers': 2,
                'neurons_per_layer': 64,
                'learning_rate': 0.001,
                'dropout_rate': 0.2,
                'l2_reg': 0.01
            }
        
        model = self.create_model(input_dim, **best_params)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
        
        history = model.fit(X_train, y_train, 
                           epochs=200, batch_size=32, 
                           validation_split=0.2, 
                           callbacks=[early_stopping, reduce_lr], 
                           verbose=0)
        
        training_time = time.time() - start_time
        
        y_train_pred = model.predict(X_train, verbose=0).ravel()
        y_test_pred = model.predict(X_test, verbose=0).ravel()
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results = {
            'model': model,
            'model_type': 'neural_network',
            'training_time': training_time,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'history': history.history,
            'best_params': best_params,
            'epochs_trained': len(history.history['loss'])
        }
        
        return results

class EvaluationMetrics:
    @staticmethod
    def calculate_expressive_power_metrics(model_results: Dict, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        train_r2 = model_results['train_r2']
        residuals = y_train - model_results['model'].predict(X_train)
        
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)
        
        return {
            'train_r2': train_r2,
            'residual_std': residual_std,
            'residual_mean': residual_mean,
            'fit_quality': train_r2 * (1 - residual_std)
        }
    
    @staticmethod
    def calculate_generalization_metrics(model_results: Dict) -> Dict:
        return {
            'test_r2': model_results['test_r2'],
            'test_mse': model_results['test_mse'],
            'test_mae': model_results['test_mae'],
            'generalization_gap': model_results['train_r2'] - model_results['test_r2']
        }
    
    @staticmethod
    def calculate_training_cost_metrics(model_results: Dict) -> Dict:
        training_time = model_results['training_time']
        
        if model_results['model_type'] == 'neural_network':
            epochs = model_results['epochs_trained']
            params = model_results['best_params']
            complexity_score = params['hidden_layers'] * params['neurons_per_layer']
        else:
            epochs = 1
            complexity_score = len(model_results['coefficients']) if model_results['coefficients'] is not None else 0
        
        return {
            'training_time': training_time,
            'epochs': epochs,
            'complexity_score': complexity_score,
            'time_per_epoch': training_time / epochs if epochs > 0 else training_time
        }
    
    @staticmethod
    def calculate_interpretability_metrics(model_results: Dict, feature_names: List[str] = None) -> Dict:
        if model_results['model_type'] == 'neural_network':
            return {
                'interpretability_score': 0.2,
                'feature_importance_available': False,
                'coefficients_available': False,
                'model_complexity': 'high'
            }
        else:
            coefficients = model_results['coefficients']
            if coefficients is not None:
                feature_importance = np.abs(coefficients)
                most_important_idx = np.argmax(feature_importance)
                
                return {
                    'interpretability_score': 0.9,
                    'feature_importance_available': True,
                    'coefficients_available': True,
                    'model_complexity': 'low',
                    'n_features': len(coefficients),
                    'most_important_feature_idx': most_important_idx,
                    'most_important_feature_name': feature_names[most_important_idx] if feature_names else f"feature_{most_important_idx}",
                    'most_important_feature_weight': coefficients[most_important_idx]
                }
            else:
                return {
                    'interpretability_score': 0.7,
                    'feature_importance_available': False,
                    'coefficients_available': False,
                    'model_complexity': 'medium'
                }

class StatisticalAnalyzer:
    @staticmethod
    def paired_t_test(results1: List[float], results2: List[float]) -> Dict:
        if len(results1) != len(results2):
            raise ValueError("Result lists must have the same length")
        
        t_stat, p_value = ttest_rel(results1, results2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_diff': np.mean(results1) - np.mean(results2),
            'effect_size': np.abs(np.mean(results1) - np.mean(results2)) / np.std(np.array(results1) - np.array(results2))
        }
    
    @staticmethod
    def wilcoxon_signed_rank_test(results1: List[float], results2: List[float]) -> Dict:
        if len(results1) != len(results2):
            raise ValueError("Result lists must have the same length")
        
        stat, p_value = wilcoxon(results1, results2)
        
        return {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'median_diff': np.median(results1) - np.median(results2)
        }
    
    @staticmethod
    def bootstrap_confidence_interval(data: List[float], confidence_level: float = 0.95, n_bootstrap: int = 1000) -> Dict:
        n = len(data)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return {
            'mean': np.mean(data),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level,
            'bootstrap_means': bootstrap_means
        }

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_generator = DataGenerator(config)
        self.preprocessor = DataPreprocessor(config)
        self.linear_models = LinearRegressionModels(config)
        self.nn_models = NeuralNetworkModels(config)
        self.evaluator = EvaluationMetrics()
        self.analyzer = StatisticalAnalyzer()
        self.results = {}
        
    def run_single_experiment(self, X: np.ndarray, y: np.ndarray, metadata: Dict) -> Dict:
        preprocessed_data = self.preprocessor.preprocess_synthetic_data(X, y)
        
        X_train, X_test = preprocessed_data['X_train'], preprocessed_data['X_test']
        y_train, y_test = preprocessed_data['y_train'], preprocessed_data['y_test']
        
        linear_results = {}
        for model_type in ['ols', 'ridge', 'lasso']:
            result = self.linear_models.train_and_evaluate(X_train, y_train, X_test, y_test, model_type)
            linear_results[model_type] = result
        
        nn_result = self.nn_models.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        experiment_results = {
            'metadata': metadata,
            'linear_models': linear_results,
            'neural_network': nn_result,
            'preprocessing': preprocessed_data
        }
        
        return experiment_results
    
    def run_synthetic_experiments(self) -> Dict:
        synthetic_datasets = self.data_generator.generate_synthetic_datasets()
        results = {}
        
        for dataset_key, dataset_list in synthetic_datasets.items():
            results[dataset_key] = []
            
            for X, y, metadata in dataset_list:
                experiment_result = self.run_single_experiment(X, y, metadata)
                results[dataset_key].append(experiment_result)
        
        return results
    
    def run_california_housing_experiment(self) -> Dict:
        X, y, metadata = self.data_generator.load_california_housing()
        return self.run_single_experiment(X, y, metadata)
    
    def evaluate_all_results(self, results: Dict) -> Dict:
        evaluated_results = {}
        
        for experiment_key, experiment_list in results.items():
            evaluated_results[experiment_key] = []
            
            for experiment in experiment_list:
                evaluated_experiment = {
                    'metadata': experiment['metadata'],
                    'linear_models': {},
                    'neural_network': {}
                }
                
                for model_type, model_result in experiment['linear_models'].items():
                    expressive_power = self.evaluator.calculate_expressive_power_metrics(
                        model_result, experiment['preprocessing']['X_train'], experiment['preprocessing']['y_train'])
                    generalization = self.evaluator.calculate_generalization_metrics(model_result)
                    training_cost = self.evaluator.calculate_training_cost_metrics(model_result)
                    interpretability = self.evaluator.calculate_interpretability_metrics(model_result)
                    
                    evaluated_experiment['linear_models'][model_type] = {
                        'results': model_result,
                        'expressive_power': expressive_power,
                        'generalization': generalization,
                        'training_cost': training_cost,
                        'interpretability': interpretability
                    }
                
                nn_result = experiment['neural_network']
                nn_expressive_power = self.evaluator.calculate_expressive_power_metrics(
                    nn_result, experiment['preprocessing']['X_train'], experiment['preprocessing']['y_train'])
                nn_generalization = self.evaluator.calculate_generalization_metrics(nn_result)
                nn_training_cost = self.evaluator.calculate_training_cost_metrics(nn_result)
                nn_interpretability = self.evaluator.calculate_interpretability_metrics(nn_result)
                
                evaluated_experiment['neural_network'] = {
                    'results': nn_result,
                    'expressive_power': nn_expressive_power,
                    'generalization': nn_generalization,
                    'training_cost': nn_training_cost,
                    'interpretability': nn_interpretability
                }
                
                evaluated_results[experiment_key].append(evaluated_experiment)
        
        return evaluated_results
    
    def compare_models(self, evaluated_results: Dict) -> Dict:
        comparison_results = {}
        
        for experiment_key, experiment_list in evaluated_results.items():
            comparison_results[experiment_key] = []
            
            for experiment in experiment_list:
                comparison = {
                    'metadata': experiment['metadata'],
                    'expressive_power': {},
                    'generalization': {},
                    'training_cost': {},
                    'interpretability': {}
                }
                
                linear_ols = experiment['linear_models']['ols']
                linear_ridge = experiment['linear_models']['ridge']
                linear_lasso = experiment['linear_models']['lasso']
                neural_net = experiment['neural_network']
                
                comparison['expressive_power'] = {
                    'linear_ols': linear_ols['expressive_power']['train_r2'],
                    'linear_ridge': linear_ridge['expressive_power']['train_r2'],
                    'linear_lasso': linear_lasso['expressive_power']['train_r2'],
                    'neural_network': neural_net['expressive_power']['train_r2']
                }
                
                comparison['generalization'] = {
                    'linear_ols': linear_ols['generalization']['test_r2'],
                    'linear_ridge': linear_ridge['generalization']['test_r2'],
                    'linear_lasso': linear_lasso['generalization']['test_r2'],
                    'neural_network': neural_net['generalization']['test_r2']
                }
                
                comparison['training_cost'] = {
                    'linear_ols': linear_ols['training_cost']['training_time'],
                    'linear_ridge': linear_ridge['training_cost']['training_time'],
                    'linear_lasso': linear_lasso['training_cost']['training_time'],
                    'neural_network': neural_net['training_cost']['training_time']
                }
                
                comparison['interpretability'] = {
                    'linear_ols': linear_ols['interpretability']['interpretability_score'],
                    'linear_ridge': linear_ridge['interpretability']['interpretability_score'],
                    'linear_lasso': linear_lasso['interpretability']['interpretability_score'],
                    'neural_network': neural_net['interpretability']['interpretability_score']
                }
                
                comparison_results[experiment_key].append(comparison)
        
        return comparison_results
    
    def run_statistical_analysis(self, comparison_results: Dict) -> Dict:
        statistical_results = {}
        
        for dimension in ['expressive_power', 'generalization', 'training_cost', 'interpretability']:
            statistical_results[dimension] = {}
            
            linear_ols_scores = []
            linear_ridge_scores = []
            linear_lasso_scores = []
            neural_net_scores = []
            
            for experiment_key, experiment_list in comparison_results.items():
                for experiment in experiment_list:
                    linear_ols_scores.append(experiment[dimension]['linear_ols'])
                    linear_ridge_scores.append(experiment[dimension]['linear_ridge'])
                    linear_lasso_scores.append(experiment[dimension]['linear_lasso'])
                    neural_net_scores.append(experiment[dimension]['neural_network'])
            
            ols_vs_nn = self.analyzer.paired_t_test(linear_ols_scores, neural_net_scores)
            ridge_vs_nn = self.analyzer.paired_t_test(linear_ridge_scores, neural_net_scores)
            lasso_vs_nn = self.analyzer.paired_t_test(linear_lasso_scores, neural_net_scores)
            
            ols_ci = self.analyzer.bootstrap_confidence_interval(linear_ols_scores)
            ridge_ci = self.analyzer.bootstrap_confidence_interval(linear_ridge_scores)
            lasso_ci = self.analyzer.bootstrap_confidence_interval(linear_lasso_scores)
            nn_ci = self.analyzer.bootstrap_confidence_interval(neural_net_scores)
            
            statistical_results[dimension] = {
                'linear_ols_vs_neural_network': ols_vs_nn,
                'linear_ridge_vs_neural_network': ridge_vs_nn,
                'linear_lasso_vs_neural_network': lasso_vs_nn,
                'linear_ols_ci': ols_ci,
                'linear_ridge_ci': ridge_ci,
                'linear_lasso_ci': lasso_ci,
                'neural_network_ci': nn_ci
            }
        
        return statistical_results
    
    def run_full_experiment(self) -> Dict:
        print("Running synthetic experiments...")
        synthetic_results = self.run_synthetic_experiments()
        
        print("Running California Housing experiment...")
        california_results = self.run_california_housing_experiment()
        
        all_results = {
            'synthetic': synthetic_results,
            'california_housing': california_results
        }
        
        print("Evaluating results...")
        evaluated_results = {
            'synthetic': self.evaluate_all_results(synthetic_results),
            'california_housing': self.evaluate_all_results({'california': [california_results]})
        }
        
        print("Comparing models...")
        comparison_results = {
            'synthetic': self.compare_models(evaluated_results['synthetic']),
            'california_housing': self.compare_models(evaluated_results['california_housing'])
        }
        
        print("Running statistical analysis...")
        statistical_results = {
            'synthetic': self.run_statistical_analysis(comparison_results['synthetic']),
            'california_housing': self.run_statistical_analysis(comparison_results['california_housing'])
        }
        
        final_results = {
            'raw_results': all_results,
            'evaluated_results': evaluated_results,
            'comparison_results': comparison_results,
            'statistical_analysis': statistical_results
        }
        
        self.results = final_results
        return final_results

if __name__ == "__main__":
    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    results = runner.run_full_experiment()
    
    print("Experiment completed successfully!")
    print(f"Results keys: {list(results.keys())}")