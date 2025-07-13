#!/usr/bin/env python3
"""
Model Comparison Script

This script compares different model configurations and architectures
to find the best performing model for student performance prediction.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from data_collection import DataCollector
from data_preprocessing import DataPreprocessor
from neural_network_model import create_model

def create_model_configs():
    """Define different model configurations to compare"""
    
    configs = {
        'shallow_wide': {
            'hidden_layers': [256, 128],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'activation': 'relu',
            'l2_reg': 0.01
        },
        'deep_narrow': {
            'hidden_layers': [64, 32, 16, 8],
            'dropout_rate': 0.4,
            'learning_rate': 0.0005,
            'activation': 'elu',
            'l2_reg': 0.005
        },
        'medium_balanced': {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.25,
            'learning_rate': 0.0008,
            'activation': 'swish',
            'l2_reg': 0.008
        },
        'high_dropout': {
            'hidden_layers': [96, 48, 24],
            'dropout_rate': 0.5,
            'learning_rate': 0.0012,
            'activation': 'relu',
            'l2_reg': 0.015
        },
        'low_l2': {
            'hidden_layers': [160, 80, 40, 20],
            'dropout_rate': 0.2,
            'learning_rate': 0.0006,
            'activation': 'elu',
            'l2_reg': 0.001
        }
    }
    
    return configs

def train_and_evaluate_model(config_name, config, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model configuration"""
    
    print(f"Training {config_name}...")
    
    # Create model
    model = create_model(
        input_shape=(X_train.shape[1],),
        hidden_layers=config['hidden_layers'],
        dropout_rate=config['dropout_rate'],
        learning_rate=config['learning_rate'],
        activation=config['activation'],
        l2_reg=config['l2_reg']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate accuracy within ±5 points
    acc_5 = np.mean(np.abs(y_test - y_pred.flatten()) <= 5) * 100
    
    # Calculate correlation
    corr = np.corrcoef(y_test, y_pred.flatten())[0, 1]
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/student_performance_model_{timestamp}.keras'
    model.save(model_path)
    
    results = {
        'config_name': config_name,
        'model_path': model_path,
        'metrics': {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'acc_5': float(acc_5),
            'corr': float(corr)
        },
        'config': config,
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']]
        }
    }
    
    return results

def compare_models():
    """Compare different model configurations"""
    
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    collector = DataCollector()
    df = collector.load_csv()
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Get model configurations
    configs = create_model_configs()
    
    # Train and evaluate all models
    results = {}
    
    for config_name, config in configs.items():
        try:
            result = train_and_evaluate_model(
                config_name, config, X_train, X_test, y_train, y_test
            )
            results[config_name] = result
            print(f"✅ {config_name}: MAE = {result['metrics']['mae']:.4f}")
        except Exception as e:
            print(f"❌ {config_name}: Error - {e}")
    
    # Find best model
    best_model = min(results.keys(), key=lambda x: results[x]['metrics']['mae'])
    
    # Create comparison summary
    comparison_summary = {
        'timestamp': datetime.now().isoformat(),
        'best_model': best_model,
        'best_mae': results[best_model]['metrics']['mae'],
        'models': results
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'reports/model_comparison_{timestamp}.json', 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    # Print summary
    print(f"\n📊 **MODEL COMPARISON RESULTS**")
    print("-" * 50)
    print(f"{'Model':<15} {'MAE':<8} {'R²':<8} {'Acc ±5':<8}")
    print("-" * 50)
    
    for config_name, result in results.items():
        metrics = result['metrics']
        print(f"{config_name:<15} {metrics['mae']:<8.4f} {metrics['r2']:<8.4f} {metrics['acc_5']:<8.2f}%")
    
    print(f"\n🏆 **Best Model**: {best_model}")
    print(f"📊 MAE: {results[best_model]['metrics']['mae']:.4f}")
    print(f"📊 R²: {results[best_model]['metrics']['r2']:.4f}")
    print(f"📊 Accuracy ±5: {results[best_model]['metrics']['acc_5']:.2f}%")
    
    # Create visualization
    create_comparison_plot(results)
    
    return results, comparison_summary

def create_comparison_plot(results):
    """Create comparison visualization"""
    
    # Prepare data for plotting
    models = list(results.keys())
    mae_values = [results[model]['metrics']['mae'] for model in models]
    r2_values = [results[model]['metrics']['r2'] for model in models]
    acc_values = [results[model]['metrics']['acc_5'] for model in models]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MAE comparison
    bars1 = axes[0].bar(models, mae_values, color='skyblue', alpha=0.7)
    axes[0].set_title('Mean Absolute Error (MAE)')
    axes[0].set_ylabel('MAE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, mae in zip(bars1, mae_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{mae:.3f}', ha='center', va='bottom')
    
    # R² comparison
    bars2 = axes[1].bar(models, r2_values, color='lightgreen', alpha=0.7)
    axes[1].set_title('R² Score')
    axes[1].set_ylabel('R²')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, r2 in zip(bars2, r2_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom')
    
    # Accuracy comparison
    bars3 = axes[2].bar(models, acc_values, color='lightcoral', alpha=0.7)
    axes[2].set_title('Accuracy ±5 Points')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, acc in zip(bars3, acc_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📈 Comparison plot saved to plots/model_comparison.png")

if __name__ == "__main__":
    compare_models() 