#!/usr/bin/env python3
"""
Hyperparameter Tuning Script

This script performs hyperparameter optimization using Keras Tuner
to find the best model configuration for student performance prediction.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
from sklearn.model_selection import train_test_split
from data_collection import DataCollector
from data_preprocessing import DataPreprocessor

class StudentPerformanceHyperModel(keras_tuner.HyperModel):
    """Hypermodel for student performance prediction"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        """Build the model with hyperparameters"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            units=hp.Int('input_units', min_value=32, max_value=256, step=32),
            activation=hp.Choice('input_activation', values=['relu', 'elu', 'swish']),
            input_shape=self.input_shape
        ))
        
        # Hidden layers
        n_layers = hp.Int('n_layers', min_value=1, max_value=4)
        
        for i in range(n_layers):
            model.add(layers.Dense(
                units=hp.Int(f'layer_{i}_units', min_value=16, max_value=128, step=16),
                activation=hp.Choice(f'layer_{i}_activation', values=['relu', 'elu', 'swish'])
            ))
            
            # Dropout
            if hp.Boolean(f'layer_{i}_dropout'):
                model.add(layers.Dropout(
                    hp.Float(f'layer_{i}_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
                ))
            
            # Batch normalization
            if hp.Boolean(f'layer_{i}_batch_norm'):
                model.add(layers.BatchNormalization())
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
            ),
            loss='mse',
            metrics=['mae']
        )
        
        return model

def tune_hyperparameters():
    """Perform hyperparameter tuning"""
    print("=" * 60)
    print("HYPERPARAMETER TUNING")
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
    
    # Create hypermodel
    hypermodel = StudentPerformanceHyperModel(input_shape=(X_train.shape[1],))
    
    # Create tuner
    tuner = keras_tuner.RandomSearch(
        hypermodel,
        objective=keras_tuner.Objective('val_mae', direction='min'),
        max_trials=20,
        executions_per_trial=2,
        directory='tuner_results',
        project_name='student_performance'
    )
    
    # Perform search
    print("Starting hyperparameter search...")
    tuner.search(
        X_train, y_train,
        epochs=50,
        validation_split=0.2,
        batch_size=32,
        verbose=1
    )
    
    # Get best model
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    
    # Evaluate best model
    test_loss, test_mae = best_model.evaluate(X_test, y_test, verbose=0)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best model
    model_path = f'models/student_performance_tuned_{timestamp}.keras'
    best_model.save(model_path)
    
    # Save hyperparameters
    hp_path = f'models/best_hyperparameters_{timestamp}.txt'
    with open(hp_path, 'w') as f:
        f.write("Best Hyperparameters:\n")
        f.write("=" * 30 + "\n")
        for param, value in best_hyperparameters.values.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nTest MAE: {test_mae:.4f}\n")
    
    # Save results summary
    results = {
        'timestamp': timestamp,
        'best_hyperparameters': best_hyperparameters.values,
        'test_mae': float(test_mae),
        'test_loss': float(test_loss),
        'model_path': model_path,
        'hyperparameters_path': hp_path
    }
    
    with open(f'reports/tuning_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Tuning completed!")
    print(f"📁 Best model saved to: {model_path}")
    print(f"📁 Hyperparameters saved to: {hp_path}")
    print(f"📊 Test MAE: {test_mae:.4f}")
    
    return best_model, best_hyperparameters, results

if __name__ == "__main__":
    tune_hyperparameters() 