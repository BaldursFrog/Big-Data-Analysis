import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.feature import hog

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from scipy.optimize import linear_sum_assignment
from scipy.stats import mode
import multiprocessing as mp

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
np.random.seed(42)
tf.random.set_seed(42)

class ExperimentConfig:
    def __init__(self):
        self.random_seeds = [42, 123, 456, 789, 999]
        self.n_clusters = 10
        self.test_size = 0.2
        self.subset_size = 5000
        
        self.kmeans_params = {
            'n_clusters': self.n_clusters,
            'init': 'k-means++',
            'max_iter': 300,
            'tol': 1e-4,
            'n_init': 10,
            'random_state': 42
        }
        
        self.deepcluster_params = {
            'input_shape': (28, 28, 1),
            'feature_dim': 64,
            'batch_size': 128,
            'epochs': 50,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'clustering_frequency': 5
        }
        
        self.feature_extraction_params = {
            'pca_components': 50,
            'hog_pixels_per_cell': (8, 8),
            'hog_cells_per_block': (2, 2),
            'hog_orientations': 9
        }

class MNISTDataLoader:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def load_mnist(self, subset_size: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)
        
        if subset_size:
            indices = np.random.choice(len(X), subset_size, replace=False)
            X, y = X[indices], y[indices]
        
        X = X.reshape(-1, 28, 28, 1) / 255.0
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test

class FeatureExtractor:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def extract_raw_pixels(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)
    
    def extract_pca_features(self, X: np.ndarray) -> np.ndarray:
        X_flat = X.reshape(X.shape[0], -1)
        pca = PCA(n_components=self.config.feature_extraction_params['pca_components'], random_state=42)
        return pca.fit_transform(X_flat)
    
    def extract_hog_features(self, X: np.ndarray) -> np.ndarray:
        hog_features = []
        for img in X:
            img_2d = img.squeeze()
            hog_feat = hog(
                img_2d,
                orientations=self.config.feature_extraction_params['hog_orientations'],
                pixels_per_cell=self.config.feature_extraction_params['hog_pixels_per_cell'],
                cells_per_block=self.config.feature_extraction_params['hog_cells_per_block'],
                block_norm='L2-Hys'
            )
            hog_features.append(hog_feat)
        return np.array(hog_features)
    
    def extract_autoencoder_features(self, X: np.ndarray, encoding_dim: int = 64) -> np.ndarray:
        input_shape = X.shape[1:]
        
        autoencoder = Sequential([
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(encoding_dim, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(np.prod(input_shape), activation='sigmoid'),
            layers.Reshape(input_shape)
        ])
        
        encoder = Model(
            inputs=autoencoder.input,
            outputs=autoencoder.layers[2].output
        )
        
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        
        X_flat = X.reshape(X.shape[0], -1)
        autoencoder.fit(
            X, X, epochs=20, batch_size=128, shuffle=True, verbose=0
        )
        
        return encoder.predict(X, verbose=0)

class KMeansClustering:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        
    def fit_predict(self, X: np.ndarray, use_mini_batch: bool = False) -> np.ndarray:
        start_time = time.time()
        
        if use_mini_batch:
            self.model = MiniBatchKMeans(
                n_clusters=self.config.kmeans_params['n_clusters'],
                batch_size=1000,
                random_state=self.config.kmeans_params['random_state']
            )
        else:
            self.model = KMeans(**self.config.kmeans_params)
        
        cluster_labels = self.model.fit_predict(X)
        training_time = time.time() - start_time
        
        return cluster_labels, training_time

class DeepCluster:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.feature_extractor = None
        self.kmeans = KMeans(n_clusters=config.n_clusters, random_state=42)
        
    def build_model(self) -> Model:
        input_shape = self.config.deepcluster_params['input_shape']
        feature_dim = self.config.deepcluster_params['feature_dim']
        
        inputs = layers.Input(shape=input_shape)
        
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(feature_dim, activation='relu')(x)
        outputs = layers.Dense(self.config.n_clusters, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        self.feature_extractor = Model(
            inputs=model.input,
            outputs=model.layers[-2].output
        )
        
        optimizer = SGD(
            learning_rate=self.config.deepcluster_params['learning_rate'],
            momentum=self.config.deepcluster_params['momentum'],
            decay=self.config.deepcluster_params['weight_decay']
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        start_time = time.time()
        
        if self.model is None:
            self.build_model()
        
        datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05
        )
        
        pseudo_labels = np.random.randint(0, self.config.n_clusters, size=X.shape[0])
        
        for epoch in range(self.config.deepcluster_params['epochs']):
            if epoch % self.config.deepcluster_params['clustering_frequency'] == 0:
                features = self.feature_extractor.predict(X, verbose=0, batch_size=64)
                pseudo_labels = self.kmeans.fit_predict(features)
            
            self.model.fit(
                datagen.flow(X, pseudo_labels, batch_size=self.config.deepcluster_params['batch_size']),
                epochs=1,
                verbose=0
            )
        
        features = self.feature_extractor.predict(X, verbose=0, batch_size=64)
        cluster_labels = self.kmeans.predict(features)
        
        training_time = time.time() - start_time
        
        return cluster_labels, training_time

class ClusteringMetrics:
    @staticmethod
    def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        
        ind = linear_sum_assignment(w.max() - w)
        acc = sum([w[i, j] for i, j in zip(ind[0], ind[1])]) / y_pred.size
        
        return acc
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> Dict[str, float]:
        metrics = {}
        
        metrics['clustering_accuracy'] = ClusteringMetrics.clustering_accuracy(y_true, y_pred)
        metrics['nmi'] = normalized_mutual_info_score(y_true, y_pred)
        metrics['ari'] = adjusted_rand_score(y_pred, y_true)
        
        if len(np.unique(y_pred)) > 1:
            metrics['silhouette_score'] = silhouette_score(X, y_pred)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, y_pred)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, y_pred)
        else:
            metrics['silhouette_score'] = 0.0
            metrics['davies_bouldin_score'] = float('inf')
            metrics['calinski_harabasz_score'] = 0.0
        
        return metrics

class Visualizer:
    @staticmethod
    def plot_cluster_samples(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, 
                            n_samples: int = 10, save_path: str = None) -> None:
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        
        for cluster_id in range(min(10, len(np.unique(y_pred)))):
            cluster_indices = np.where(y_pred == cluster_id)[0]
            
            if len(cluster_indices) > 0:
                sample_indices = np.random.choice(cluster_indices, min(n_samples, len(cluster_indices)), replace=False)
                
                for i, idx in enumerate(sample_indices[:5]):
                    if i < 5:
                        axes[0, cluster_id].imshow(X[idx].squeeze(), cmap='gray')
                        axes[0, cluster_id].set_title(f"Cluster {cluster_id}")
                        axes[0, cluster_id].axis('off')
                        
                        true_label = y_true[idx]
                        axes[1, cluster_id].text(0.5, 0.5, str(true_label), 
                                                ha='center', va='center', fontsize=12)
                        axes[1, cluster_id].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(results: Dict, save_path: str = None) -> None:
        metrics = ['clustering_accuracy', 'nmi', 'ari', 'silhouette_score']
        algorithms = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[alg][metric] for alg in algorithms]
            axes[i].bar(algorithms, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_loader = MNISTDataLoader(config)
        self.feature_extractor = FeatureExtractor(config)
        self.kmeans = KMeansClustering(config)
        self.deepcluster = DeepCluster(config)
        self.metrics_calculator = ClusteringMetrics()
        self.visualizer = Visualizer()
        self.results = {}
        
    def run_experiment(self, subset_size: int = None) -> Dict:
        X_train, X_test, y_train, y_test = self.data_loader.load_mnist(subset_size)
        
        results = {
            'dataset_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': X_train.shape[1] * X_train.shape[2],
                'n_classes': len(np.unique(y_train))
            }
        }
        
        feature_methods = ['raw_pixels', 'pca', 'hog']
        
        for method in feature_methods:
            print(f"\nExtracting {method} features...")
            
            if method == 'raw_pixels':
                X_train_features = self.feature_extractor.extract_raw_pixels(X_train)
                X_test_features = self.feature_extractor.extract_raw_pixels(X_test)
            elif method == 'pca':
                X_train_features = self.feature_extractor.extract_pca_features(X_train)
                X_test_features = self.feature_extractor.extract_pca_features(X_test)
            elif method == 'hog':
                X_train_features = self.feature_extractor.extract_hog_features(X_train)
                X_test_features = self.feature_extractor.extract_hog_features(X_test)
            
            print(f"Feature shape: {X_train_features.shape}")
            
            print(f"\nRunning K-means with {method} features...")
            kmeans_labels, kmeans_time = self.kmeans.fit_predict(X_train_features)
            kmeans_metrics = self.metrics_calculator.calculate_all_metrics(y_train, kmeans_labels, X_train_features)
            kmeans_metrics['training_time'] = kmeans_time
            
            results[f'kmeans_{method}'] = kmeans_metrics
            
            print(f"K-means ACC: {kmeans_metrics['clustering_accuracy']:.4f}, NMI: {kmeans_metrics['nmi']:.4f}")
        
        print("\nRunning DeepCluster...")
        deepcluster_labels, deepcluster_time = self.deepcluster.fit_predict(X_train)
        deepcluster_metrics = self.metrics_calculator.calculate_all_metrics(y_train, deepcluster_labels, X_train.reshape(X_train.shape[0], -1))
        deepcluster_metrics['training_time'] = deepcluster_time
        
        results['deepcluster'] = deepcluster_metrics
        
        print(f"DeepCluster ACC: {deepcluster_metrics['clustering_accuracy']:.4f}, NMI: {deepcluster_metrics['nmi']:.4f}")
        
        self.visualizer.plot_metrics_comparison(results, save_path='experiment2_mnist_metrics.png')
        
        best_kmeans_method = max([f'kmeans_{method}' for method in feature_methods], 
                                key=lambda x: results[x]['clustering_accuracy'])
        
        self.visualizer.plot_cluster_samples(
            X_train, y_train, 
            self.kmeans.model.labels_ if self.kmeans.model else kmeans_labels,
            save_path='experiment2_mnist_kmeans_samples.png'
        )
        
        self.visualizer.plot_cluster_samples(
            X_train, y_train, deepcluster_labels,
            save_path='experiment2_mnist_deepcluster_samples.png'
        )
        
        with open('experiment2_mnist_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        self.results = results
        return results

if __name__ == "__main__":
    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    results = runner.run_experiment(subset_size=config.subset_size)
    
    print("\nExperiment completed successfully!")
    print("Results saved to experiment2_mnist_results.pkl")
    print("Visualizations saved as PNG files")