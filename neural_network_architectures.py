#!/usr/bin/env python3
"""
Neural Network Architecture Comparison for Student Performance Prediction

This module provides different neural network architectures optimized for:
- 13 input features (enhanced preprocessing)
- Regression problem (exam score prediction)
- Educational data characteristics
- Real-world deployment requirements
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class ArchitectureComparison:
    """Compare different neural network architectures for student performance prediction"""
    
    def __init__(self, input_dim: int = 13):
        self.input_dim = input_dim
        self.architectures = self._define_architectures()
        self.results = {}
        
    def _define_architectures(self) -> Dict[str, Dict[str, Any]]:
        """Define different neural network architectures"""
        return {
            # 1. Simple Architecture (Baseline)
            'simple': {
                'name': 'Simple Neural Network',
                'description': 'Basic 2-layer network for quick baseline',
                'hidden_layers': [32, 16],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'l2_reg': 0.001,
                'pros': ['Fast training', 'Easy to interpret', 'Low computational cost'],
                'cons': ['Limited capacity', 'May underfit complex patterns'],
                'best_for': 'Quick prototyping and baseline performance'
            },
            
            # 2. Standard Architecture (Balanced)
            'standard': {
                'name': 'Standard Neural Network',
                'description': 'Balanced 3-layer network for general use',
                'hidden_layers': [64, 32, 16],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'l2_reg': 0.01,
                'pros': ['Good balance of capacity and speed', 'Well-tested architecture'],
                'cons': ['May not capture very complex patterns'],
                'best_for': 'General student performance prediction'
            },
            
            # 3. Deep Architecture (Enhanced)
            'deep': {
                'name': 'Deep Neural Network',
                'description': '4-layer deep network for complex patterns',
                'hidden_layers': [128, 64, 32, 16],
                'dropout_rate': 0.4,
                'learning_rate': 0.0005,
                'l2_reg': 0.01,
                'pros': ['High capacity', 'Can capture complex relationships', 'Good for 13 features'],
                'cons': ['Slower training', 'Risk of overfitting', 'More parameters'],
                'best_for': 'Complex student performance patterns with many features'
            },
            
            # 4. Wide Architecture (Feature-focused)
            'wide': {
                'name': 'Wide Neural Network',
                'description': 'Wide layers for feature interaction',
                'hidden_layers': [256, 128, 64],
                'dropout_rate': 0.3,
                'learning_rate': 0.0008,
                'l2_reg': 0.005,
                'pros': ['Good feature interaction', 'High expressiveness'],
                'cons': ['Computationally expensive', 'Risk of overfitting'],
                'best_for': 'When feature interactions are important'
            },
            
            # 5. Residual Architecture (Advanced)
            'residual': {
                'name': 'Residual Neural Network',
                'description': 'Residual connections for better gradient flow',
                'hidden_layers': [64, 64, 32, 32, 16],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'l2_reg': 0.01,
                'pros': ['Better gradient flow', 'Easier training of deep networks'],
                'cons': ['More complex', 'Harder to interpret'],
                'best_for': 'Deep networks with potential gradient issues'
            },
            
            # 6. Attention-based Architecture (Modern)
            'attention': {
                'name': 'Attention-based Network',
                'description': 'Self-attention mechanism for feature importance',
                'hidden_layers': [64, 32, 16],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'l2_reg': 0.01,
                'pros': ['Automatic feature weighting', 'Interpretable attention'],
                'cons': ['More complex', 'Computational overhead'],
                'best_for': 'When feature importance varies by student'
            },
            
            # 7. Ensemble-ready Architecture (Production)
            'ensemble': {
                'name': 'Ensemble-ready Network',
                'description': 'Optimized for ensemble methods',
                'hidden_layers': [96, 48, 24],
                'dropout_rate': 0.35,
                'learning_rate': 0.0008,
                'l2_reg': 0.015,
                'pros': ['Good for ensembles', 'Balanced performance'],
                'cons': ['Not the best standalone'],
                'best_for': 'Production systems with ensemble methods'
            }
        }
    
    def build_architecture(self, arch_name: str) -> tf.keras.Model:
        """Build a specific neural network architecture"""
        if arch_name not in self.architectures:
            raise ValueError(f"Architecture '{arch_name}' not found")
        
        arch = self.architectures[arch_name]
        
        if arch_name == 'residual':
            return self._build_residual_network(arch)
        elif arch_name == 'attention':
            return self._build_attention_network(arch)
        else:
            return self._build_standard_network(arch)
    
    def _build_standard_network(self, arch: Dict[str, Any]) -> tf.keras.Model:
        """Build standard feedforward neural network"""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            arch['hidden_layers'][0],
            activation='relu',
            input_shape=(self.input_dim,),
            kernel_regularizer=tf.keras.regularizers.l2(arch['l2_reg'])
        ))
        model.add(layers.Dropout(arch['dropout_rate']))
        model.add(layers.BatchNormalization())
        
        # Hidden layers
        for units in arch['hidden_layers'][1:]:
            model.add(layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(arch['l2_reg'])
            ))
            model.add(layers.Dropout(arch['dropout_rate']))
            model.add(layers.BatchNormalization())
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=arch['learning_rate'])
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
        
        return model
    
    def _build_residual_network(self, arch: Dict[str, Any]) -> tf.keras.Model:
        """Build residual neural network"""
        inputs = tf.keras.Input(shape=(self.input_dim,))
        
        # Initial dense layer
        x = layers.Dense(arch['hidden_layers'][0], activation='relu')(inputs)
        x = layers.Dropout(arch['dropout_rate'])(x)
        x = layers.BatchNormalization()(x)
        
        # Residual blocks
        for i, units in enumerate(arch['hidden_layers'][1:], 1):
            # Residual connection
            residual = x
            
            # Main path
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(arch['dropout_rate'])(x)
            x = layers.BatchNormalization()(x)
            
            # Add residual connection if dimensions match
            if residual.shape[-1] == units:
                x = layers.Add()([x, residual])
        
        # Output layer
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=arch['learning_rate'])
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
        
        return model
    
    def _build_attention_network(self, arch: Dict[str, Any]) -> tf.keras.Model:
        """Build attention-based neural network"""
        inputs = tf.keras.Input(shape=(self.input_dim,))
        
        # Feature embedding
        x = layers.Dense(arch['hidden_layers'][0], activation='relu')(inputs)
        x = layers.Dropout(arch['dropout_rate'])(x)
        x = layers.BatchNormalization()(x)
        
        # Self-attention mechanism
        attention_weights = layers.Dense(arch['hidden_layers'][0], activation='softmax')(x)
        attended_features = layers.Multiply()([x, attention_weights])
        
        # Continue with standard layers
        for units in arch['hidden_layers'][1:]:
            x = layers.Dense(units, activation='relu')(attended_features)
            x = layers.Dropout(arch['dropout_rate'])(x)
            x = layers.BatchNormalization()(x)
            attended_features = x
        
        # Output layer
        outputs = layers.Dense(1, activation='linear')(attended_features)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=arch['learning_rate'])
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
        
        return model
    
    def compare_architectures(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Compare all architectures on the dataset"""
        print("=" * 80)
        print("NEURAL NETWORK ARCHITECTURE COMPARISON")
        print("=" * 80)
        
        results = {}
        
        for arch_name in self.architectures.keys():
            print(f"\n🏗️  Testing {self.architectures[arch_name]['name']}...")
            
            try:
                # Build and train model
                model = self.build_architecture(arch_name)
                
                # Training configuration
                training_config = {
                    'epochs': 30,  # Quick comparison
                    'batch_size': 64,
                    'validation_split': 0.2,
                    'early_stopping_patience': 8,
                    'lr_scheduler_patience': 5
                }
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_split=training_config['validation_split'],
                    epochs=training_config['epochs'],
                    batch_size=training_config['batch_size'],
                    callbacks=[
                        callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=training_config['early_stopping_patience'],
                            restore_best_weights=True,
                            verbose=0
                        ),
                        callbacks.ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=training_config['lr_scheduler_patience'],
                            verbose=0
                        )
                    ],
                    verbose=0
                )
                
                # Evaluate model
                y_pred = model.predict(X_test, verbose=0)
                y_pred_flat = y_pred.flatten()
                
                # Calculate metrics
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred_flat),
                    'mae': mean_absolute_error(y_test, y_pred_flat),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_flat)),
                    'r2': r2_score(y_test, y_pred_flat),
                    'mape': np.mean(np.abs((y_test - y_pred_flat) / y_test)) * 100,
                    'grade_accuracy_5': np.mean(np.abs(y_test - y_pred_flat) <= 5) * 100,
                    'grade_accuracy_10': np.mean(np.abs(y_test - y_pred_flat) <= 10) * 100,
                    'correlation': np.corrcoef(y_test, y_pred_flat)[0, 1],
                    'params': model.count_params(),
                    'final_val_loss': min(history.history['val_loss'])
                }
                
                results[arch_name] = {
                    'metrics': metrics,
                    'history': history.history,
                    'model': model
                }
                
                print(f"✅ {self.architectures[arch_name]['name']} completed")
                print(f"   R² Score: {metrics['r2']:.4f}")
                print(f"   MAE: {metrics['mae']:.4f}")
                print(f"   Grade Accuracy (±5): {metrics['grade_accuracy_5']:.1f}%")
                print(f"   Parameters: {metrics['params']:,}")
                
            except Exception as e:
                print(f"❌ Error with {arch_name}: {str(e)}")
                results[arch_name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def plot_comparison(self, save_path: str = None) -> None:
        """Plot comparison of all architectures"""
        if not self.results:
            print("No results to plot. Run compare_architectures() first.")
            return
        
        # Prepare data for plotting
        arch_names = []
        r2_scores = []
        mae_scores = []
        grade_accuracies = []
        param_counts = []
        
        for arch_name, result in self.results.items():
            if 'metrics' in result:
                arch_names.append(self.architectures[arch_name]['name'])
                r2_scores.append(result['metrics']['r2'])
                mae_scores.append(result['metrics']['mae'])
                grade_accuracies.append(result['metrics']['grade_accuracy_5'])
                param_counts.append(result['metrics']['params'])
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² Score comparison
        bars1 = axes[0, 0].bar(arch_names, r2_scores, color='skyblue')
        axes[0, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, r2_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # MAE comparison
        bars2 = axes[0, 1].bar(arch_names, mae_scores, color='lightcoral')
        axes[0, 1].set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, mae_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Grade Accuracy comparison
        bars3 = axes[1, 0].bar(arch_names, grade_accuracies, color='lightgreen')
        axes[1, 0].set_title('Grade Accuracy (±5 points) Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, grade_accuracies):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom')
        
        # Parameter count comparison
        bars4 = axes[1, 1].bar(arch_names, param_counts, color='gold')
        axes[1, 1].set_title('Model Complexity (Parameters)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Number of Parameters')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, param_counts):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                           f'{value:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_recommendations(self) -> Dict[str, List[str]]:
        """Get recommendations based on different criteria"""
        if not self.results:
            return {}
        
        # Sort architectures by different metrics
        valid_results = {k: v for k, v in self.results.items() if 'metrics' in v}
        
        # Best R² score
        best_r2 = max(valid_results.items(), key=lambda x: x[1]['metrics']['r2'])
        
        # Best MAE
        best_mae = min(valid_results.items(), key=lambda x: x[1]['metrics']['mae'])
        
        # Best grade accuracy
        best_accuracy = max(valid_results.items(), key=lambda x: x[1]['metrics']['grade_accuracy_5'])
        
        # Most efficient (best R² per parameter)
        efficiency = {k: v['metrics']['r2'] / v['metrics']['params'] for k, v in valid_results.items()}
        best_efficiency = max(efficiency.items(), key=lambda x: x[1])
        
        return {
            'best_overall_performance': [best_r2[0], f"R²: {best_r2[1]['metrics']['r2']:.4f}"],
            'best_prediction_accuracy': [best_mae[0], f"MAE: {best_mae[1]['metrics']['mae']:.4f}"],
            'best_grade_accuracy': [best_accuracy[0], f"Accuracy: {best_accuracy[1]['metrics']['grade_accuracy_5']:.1f}%"],
            'most_efficient': [best_efficiency[0], f"Efficiency: {best_efficiency[1]:.6f}"],
            'recommended_for_production': self._get_production_recommendation(valid_results)
        }
    
    def _get_production_recommendation(self, valid_results: Dict) -> List[str]:
        """Get production recommendation based on multiple factors"""
        # Score each architecture
        scores = {}
        for arch_name, result in valid_results.items():
            score = 0
            metrics = result['metrics']
            
            # R² score (40% weight)
            score += metrics['r2'] * 0.4
            
            # Grade accuracy (30% weight)
            score += (metrics['grade_accuracy_5'] / 100) * 0.3
            
            # Efficiency (20% weight)
            efficiency = metrics['r2'] / metrics['params']
            score += efficiency * 1000000 * 0.2  # Scale up for readability
            
            # Stability (10% weight) - lower MAE is better
            score += (1 - metrics['mae'] / 100) * 0.1
            
            scores[arch_name] = score
        
        best_production = max(scores.items(), key=lambda x: x[1])
        return [best_production[0], f"Score: {best_production[1]:.4f}"]

def main():
    """Test and compare different neural network architectures"""
    from data_collection import DataCollector
    from data_preprocessing import DataPreprocessor
    
    print("=" * 80)
    print("NEURAL NETWORK ARCHITECTURE ANALYSIS")
    print("=" * 80)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    collector = DataCollector()
    df = collector.load_csv()
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)
    
    print(f"Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]} dimensions")
    
    # Initialize architecture comparison
    comparison = ArchitectureComparison(input_dim=X_train.shape[1])
    
    # Compare all architectures
    results = comparison.compare_architectures(X_train, y_train, X_test, y_test)
    
    # Get recommendations
    recommendations = comparison.get_recommendations()
    
    # Print recommendations
    print("\n" + "=" * 80)
    print("🏆 ARCHITECTURE RECOMMENDATIONS")
    print("=" * 80)
    
    for criterion, (arch_name, metric) in recommendations.items():
        arch_info = comparison.architectures[arch_name]
        print(f"\n{criterion.replace('_', ' ').title()}:")
        print(f"  Architecture: {arch_info['name']}")
        print(f"  Metric: {metric}")
        print(f"  Description: {arch_info['description']}")
        print(f"  Best for: {arch_info['best_for']}")
        print(f"  Pros: {', '.join(arch_info['pros'])}")
        print(f"  Cons: {', '.join(arch_info['cons'])}")
    
    # Plot comparison
    print("\n📊 Generating comparison plots...")
    comparison.plot_comparison('plots/architecture_comparison.png')
    
    print("\n✅ Architecture comparison completed!")
    print("📁 Results saved to plots/architecture_comparison.png")

if __name__ == "__main__":
    main() 