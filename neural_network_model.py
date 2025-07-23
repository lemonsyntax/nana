#!/usr/bin/env python3
"""
Deep Neural Network Model for Student Performance Prediction

This module provides:
1. Deep neural network architecture (4-layer network)
2. Advanced training with early stopping and learning rate scheduling
3. Comprehensive model evaluation and performance metrics
4. Model saving and loading
5. Feature importance analysis for deep networks
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Dict, Any, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow deprecation warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')

class StudentPerformanceModel:
    """Deep neural network model for student performance regression (exam score prediction)"""
    
    def __init__(self, input_dim: int):
        """
        Initialize the model
        
        Args:
            input_dim: Number of input features
        """
        self.input_dim = input_dim
        self.model = None
        self.history = None
        self.is_trained = False
        self.feature_names = []
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def build_model(self, architecture: Dict[str, Any] = None) -> tf.keras.Model:
        """
        Build the deep neural network model for student performance regression
        
        Args:
            architecture: Dictionary with model configuration (optional)
                - hidden_layers: List of layer sizes
                - dropout_rate: Dropout rate for regularization
                - learning_rate: Learning rate for optimizer
                - activation: Activation function for hidden layers
                - output_activation: Activation function for output layer
                - l2_reg: L2 regularization strength
        
        Returns:
            Compiled Keras model
        """
        if architecture is None:
            architecture = {
                'hidden_layers': [128, 64, 32, 16],
                'dropout_rate': 0.4,
                'learning_rate': 0.0005,
                'activation': 'relu',
                'output_activation': 'linear',
                'l2_reg': 0.01
            }
        model = tf.keras.Sequential()
        model.add(layers.Dense(
            architecture['hidden_layers'][0],
            activation=architecture['activation'],
            input_shape=(self.input_dim,),
            kernel_regularizer=tf.keras.regularizers.l2(architecture.get('l2_reg', 0.01))
        ))
        model.add(layers.Dropout(architecture['dropout_rate']))
        model.add(layers.BatchNormalization())
        for i, units in enumerate(architecture['hidden_layers'][1:], 1):
            model.add(layers.Dense(
                units,
                activation=architecture['activation'],
                kernel_regularizer=tf.keras.regularizers.l2(architecture.get('l2_reg', 0.01))
            ))
            model.add(layers.Dropout(architecture['dropout_rate']))
            model.add(layers.BatchNormalization())
        model.add(layers.Dense(1, activation=architecture['output_activation']))
        loss = 'huber'
        metrics = ['mae', 'mse']
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=architecture['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              training_config: Dict[str, Any] = None) -> tf.keras.callbacks.History:
        """
        Train the deep neural network model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            training_config: Training configuration
                - epochs: Number of training epochs
                - batch_size: Batch size
                - validation_split: Fraction for validation
                - early_stopping_patience: Patience for early stopping
                - lr_scheduler_patience: Patience for learning rate reduction
                - min_delta: Minimum improvement threshold
        
        Returns:
            Training history
        """
        if self.model is None:
            # Model should be built before calling train method
            raise ValueError("Model must be built before training. Call build_model() first.")
        
        if training_config is None:
            training_config = {
                'epochs': 150,  # Optimal epochs for deep network
                'batch_size': 64,  # Optimal batch size for stability
                'validation_split': 0.2,
                'early_stopping_patience': 20,  # Patience for deep model convergence
                'lr_scheduler_patience': 15,  # Patience for learning rate scheduling
                'min_delta': 0.001  # Minimum improvement threshold
            }
        
        # Callbacks
        callbacks_list = []
        
        # Early stopping with minimum delta
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=training_config['early_stopping_patience'],
            min_delta=training_config.get('min_delta', 0.001),
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # Learning rate scheduler for deep network
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # Aggressive reduction for deep networks
            patience=training_config['lr_scheduler_patience'],
            min_lr=1e-8,  # Very low minimum learning rate
            verbose=1
        )
        callbacks_list.append(lr_scheduler)
        
        # Model checkpoint for deep network
        checkpoint = callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks_list.append(checkpoint)
        
        # Training
        if X_val is not None and y_val is not None:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=training_config['epochs'],
                batch_size=training_config['batch_size'],
                callbacks=callbacks_list,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                validation_split=training_config['validation_split'],
                epochs=training_config['epochs'],
                batch_size=training_config['batch_size'],
                callbacks=callbacks_list,
                verbose=1
            )
        
        self.is_trained = True
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model is None:
            raise ValueError("Model is not built or loaded")
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate deep neural network regression performance
        
        Args:
            X_test, y_test: Test data
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        y_pred = self.predict(X_test)
        metrics = {}
        y_pred_flat = y_pred.flatten()
        metrics['mse'] = mean_squared_error(y_test, y_pred_flat)
        metrics['mae'] = mean_absolute_error(y_test, y_pred_flat)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_test, y_pred_flat)
        metrics['mape'] = np.mean(np.abs((y_test - y_pred_flat) / y_test)) * 100
        grade_accuracy = np.mean(np.abs(y_test - y_pred_flat) <= 5) * 100
        metrics['grade_accuracy_5'] = grade_accuracy
        grade_accuracy_10 = np.mean(np.abs(y_test - y_pred_flat) <= 10) * 100
        metrics['grade_accuracy_10'] = grade_accuracy_10
        correlation = np.corrcoef(y_test, y_pred_flat)[0, 1]
        metrics['correlation'] = correlation
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot deep neural network training history (regression)
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            print("No training history available")
            return
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        metric_name = 'mae'
        axes[1].plot(self.history.history[metric_name], label=f'Training {metric_name.upper()}')
        axes[1].plot(self.history.history[f'val_{metric_name}'], label=f'Validation {metric_name.upper()}')
        axes[1].set_title(f'Model {metric_name.upper()}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name.upper())
        axes[1].legend()
        axes[1].grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        save_path: Optional[str] = None) -> None:
        """
        Plot deep neural network predictions vs actual values (regression)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        y_pred = y_pred.flatten()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Grades')
        axes[0, 0].set_ylabel('Predicted Grades')
        axes[0, 0].set_title('Actual vs Predicted Grades')
        axes[0, 0].grid(True)
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Grades')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True)
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True)
        axes[1, 1].hist(y_true, bins=30, alpha=0.7, label='Actual', edgecolor='black')
        axes[1, 1].hist(y_pred, bins=30, alpha=0.7, label='Predicted', edgecolor='black')
        axes[1, 1].set_xlabel('Grades')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Grades')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance for deep neural network
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary with feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.model is None:
            raise ValueError("Model is not built or loaded")
        # Get weights from the first layer
        weights = self.model.layers[0].get_weights()[0]
        
        # Calculate importance as the sum of absolute weights
        importance = {}
        for i, feature in enumerate(feature_names):
            if i < weights.shape[0]:
                importance[feature] = np.sum(np.abs(weights[i, :]))
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        importance = {k: v/total_importance for k, v in importance.items()}
        
        return importance
    
    def save_model(self, model_path: str = 'student_performance_model.keras') -> None:
        """
        Save the trained deep neural network model
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = 'student_performance_model.keras') -> None:
        """
        Load a trained deep neural network model
        
        Args:
            model_path: Path to the saved model
        """
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.is_trained = True
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def get_model_summary(self) -> str:
        """
        Get deep neural network architecture summary
        
        Returns:
            String representation of model architecture
        """
        if self.model is None:
            return "Model not built yet"
        
        # Capture model summary
        from io import StringIO
        summary_io = StringIO()
        self.model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        summary = summary_io.getvalue()
        summary_io.close()
        
        return summary

def main():
    """Test the deep neural network model (regression only)"""
    from data_collection import DataCollector
    from data_preprocessing import DataPreprocessor
    print("=" * 60)
    print("DEEP NEURAL NETWORK MODEL TEST")
    print("=" * 60)
    collector = DataCollector()
    df = collector.load_csv()
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)
    model = StudentPerformanceModel(input_dim=X_train.shape[1])
    architecture = {
        'hidden_layers': [128, 64, 32, 16],
        'dropout_rate': 0.4,
        'learning_rate': 0.0005,
        'activation': 'relu',
        'output_activation': 'linear',
        'l2_reg': 0.01
    }
    model.build_model(architecture)
    training_config = {
        'epochs': 50,
        'batch_size': 64,
        'validation_split': 0.2,
        'early_stopping_patience': 10,
        'lr_scheduler_patience': 8,
        'min_delta': 0.001
    }
    history = model.train(X_train, y_train, training_config=training_config)
    metrics = model.evaluate(X_test, y_test)
    print(f"\nDeep Neural Network Performance:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    feature_names = preprocessor.get_feature_names()
    importance = model.get_feature_importance(feature_names)
    print(f"\nTop 10 Most Important Features (Deep Network):")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, score) in enumerate(sorted_importance[:10], 1):
        print(f"  {i}. {feature}: {score:.4f}")
    print(f"\n✅ Deep neural network model test completed successfully!")

if __name__ == "__main__":
    main() 