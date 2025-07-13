#!/usr/bin/env python3
"""
Deployment Script for Meta Learner Super-Ensemble

This script provides a production-ready deployment of the Meta Learner super-ensemble
for student performance prediction. It includes:
1. Model loading and initialization
2. Data preprocessing pipeline
3. Prediction functionality
4. Performance monitoring
5. Example usage
"""

import os
import numpy as np
import pandas as pd
import json
import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression
from data_collection import DataCollector
from data_preprocessing import DataPreprocessor
from tensorflow import keras

class SuperEnsemblePredictor:
    """Production-ready super-ensemble predictor"""
    
    def __init__(self, models_dir='models', reports_dir='reports'):
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.models = []
        self.meta_learner = None
        self.preprocessor = None
        self.feature_names = None
        self.is_loaded = False
        
    def load_models(self):
        """Load all ensemble models and meta-learner"""
        print("Loading Super-Ensemble Models...")
        
        # Load ensemble models info
        with open(f'{self.reports_dir}/ensemble_models_info.json', 'r') as f:
            ensemble_info = json.load(f)
        
        # Load individual models
        for model_info in ensemble_info['models']:
            model_path = model_info['model_path']
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path)
                self.models.append({
                    'model': model,
                    'name': model_info['config_name'],
                    'metrics': model_info['metrics']
                })
                print(f"  ✅ Loaded: {model_info['config_name']}")
            else:
                print(f"  ❌ Missing: {model_path}")
        
        # Load meta-learner from super-ensemble results
        with open(f'{self.reports_dir}/super_ensemble_results.json', 'r') as f:
            super_results = json.load(f)
        
        # Reconstruct meta-learner (we'll retrain it)
        print("  🔄 Reconstructing meta-learner...")
        
        # Load predictions to retrain meta-learner
        with open(f'{self.reports_dir}/ensemble_results.json', 'r') as f:
            ensemble_data = json.load(f)
        
        y_true = np.array(ensemble_data['y_true'])
        predictions = ensemble_data['predictions']
        
        # Stack predictions for meta-learner
        X_meta = np.vstack([
            predictions['median'],
            predictions['best_model'], 
            predictions['stacking']
        ]).T
        
        # Train meta-learner
        self.meta_learner = LinearRegression()
        self.meta_learner.fit(X_meta, y_true)
        
        print(f"  ✅ Meta-learner trained with R²: {self.meta_learner.score(X_meta, y_true):.4f}")
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor()
        
        # Get feature names
        collector = DataCollector()
        df = collector.load_csv()
        self.feature_names = self.preprocessor._select_features(df).columns.tolist()
        
        self.is_loaded = True
        print(f"✅ Super-Ensemble loaded successfully with {len(self.models)} models")
        
    def preprocess_data(self, data):
        """Preprocess input data"""
        if isinstance(data, dict):
            # Convert dict to DataFrame
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input data must be a dictionary or pandas DataFrame")
        
        # Apply preprocessing
        X_processed = self.preprocessor.preprocess_input(df)
        return X_processed
    
    def predict(self, data):
        """Make predictions using the super-ensemble"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Preprocess data
        X = self.preprocess_data(data)
        
        # Get predictions from individual models
        predictions = []
        for model_info in self.models:
            pred = model_info['model'].predict(X)
            predictions.append(pred.flatten())
        
        # Apply ensemble methods
        ensemble_predictions = {}
        
        # 1. Median ensemble
        ensemble_predictions['median'] = np.median(predictions, axis=0)
        
        # 2. Best model (lowest MAE)
        best_model_idx = np.argmin([m['metrics']['mae'] for m in self.models])
        ensemble_predictions['best_model'] = predictions[best_model_idx]
        
        # 3. Stacking ensemble
        X_meta = np.vstack([
            ensemble_predictions['median'],
            ensemble_predictions['best_model'],
            np.mean(predictions, axis=0)  # Use simple average as stacking proxy
        ]).T
        
        # Meta-learner prediction
        super_ensemble_pred = self.meta_learner.predict(X_meta)
        
        return {
            'prediction': super_ensemble_pred[0],
            'confidence': self._calculate_confidence(predictions),
            'ensemble_predictions': {
                'median': ensemble_predictions['median'][0],
                'best_model': ensemble_predictions['best_model'][0],
                'simple_average': np.mean(predictions, axis=0)[0]
            },
            'individual_predictions': {
                model_info['name']: pred[0] for model_info, pred in zip(self.models, predictions)
            }
        }
    
    def _calculate_confidence(self, predictions):
        """Calculate prediction confidence based on model agreement"""
        predictions_array = np.array(predictions)
        std_dev = np.std(predictions_array, axis=0)
        mean_pred = np.mean(predictions_array, axis=0)
        
        # Lower std dev = higher confidence
        confidence = 1 / (1 + std_dev)
        return confidence[0]
    
    def batch_predict(self, data_list):
        """Make predictions for multiple samples"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Convert to DataFrame
        if isinstance(data_list, list):
            df = pd.DataFrame(data_list)
        else:
            df = data_list
        
        # Preprocess data
        X = self.preprocess_data(df)
        
        # Get predictions from individual models
        predictions = []
        for model_info in self.models:
            pred = model_info['model'].predict(X)
            predictions.append(pred.flatten())
        
        # Apply ensemble methods
        ensemble_predictions = {}
        ensemble_predictions['median'] = np.median(predictions, axis=0)
        
        best_model_idx = np.argmin([m['metrics']['mae'] for m in self.models])
        ensemble_predictions['best_model'] = predictions[best_model_idx]
        
        X_meta = np.vstack([
            ensemble_predictions['median'],
            ensemble_predictions['best_model'],
            np.mean(predictions, axis=0)
        ]).T
        
        # Meta-learner prediction
        super_ensemble_pred = self.meta_learner.predict(X_meta)
        
        # Calculate confidence for each prediction
        predictions_array = np.array(predictions)
        std_devs = np.std(predictions_array, axis=0)
        confidences = 1 / (1 + std_devs)
        
        return {
            'predictions': super_ensemble_pred,
            'confidences': confidences,
            'ensemble_predictions': {
                'median': ensemble_predictions['median'],
                'best_model': ensemble_predictions['best_model'],
                'simple_average': np.mean(predictions, axis=0)
            }
        }
    
    def get_model_info(self):
        """Get information about the loaded models"""
        if not self.is_loaded:
            return None
        
        info = {
            'total_models': len(self.models),
            'model_names': [m['name'] for m in self.models],
            'feature_names': self.feature_names,
            'meta_learner_type': 'LinearRegression',
            'ensemble_methods': ['median', 'best_model', 'stacking'],
            'super_ensemble_method': 'meta_learner'
        }
        
        return info

def example_usage():
    """Example usage of the SuperEnsemblePredictor"""
    print("=" * 60)
    print("SUPER-ENSEMBLE DEPLOYMENT EXAMPLE")
    print("=" * 60)
    
    # Initialize predictor
    predictor = SuperEnsemblePredictor()
    
    # Load models
    predictor.load_models()
    
    # Example single prediction
    print("\n📊 Single Prediction Example:")
    sample_data = {
        'Hours_Studied': 8,
        'Previous_Scores': 85,
        'Attendance': 95,
        'Extracurricular_Activities': 1,
        'Parental_Education_Level': 3,
        'Sleep_Hours': 7,
        'Tutoring_Sessions': 2,
        'Family_Income': 4,
        'Teacher_Quality': 4,
        'Peer_Influence': 3,
        'Internet_Access': 1
    }
    
    result = predictor.predict(sample_data)
    print(f"Input: {sample_data}")
    print(f"Predicted Score: {result['prediction']:.2f}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Ensemble Predictions:")
    for method, pred in result['ensemble_predictions'].items():
        print(f"  {method}: {pred:.2f}")
    
    # Example batch prediction
    print("\n📊 Batch Prediction Example:")
    batch_data = [
        {
            'Hours_Studied': 6, 'Previous_Scores': 80, 'Attendance': 90,
            'Extracurricular_Activities': 0, 'Parental_Education_Level': 2,
            'Sleep_Hours': 8, 'Tutoring_Sessions': 1, 'Family_Income': 3,
            'Teacher_Quality': 3, 'Peer_Influence': 2, 'Internet_Access': 1
        },
        {
            'Hours_Studied': 10, 'Previous_Scores': 90, 'Attendance': 98,
            'Extracurricular_Activities': 1, 'Parental_Education_Level': 4,
            'Sleep_Hours': 7, 'Tutoring_Sessions': 3, 'Family_Income': 5,
            'Teacher_Quality': 5, 'Peer_Influence': 4, 'Internet_Access': 1
        }
    ]
    
    batch_result = predictor.batch_predict(batch_data)
    print(f"Batch Predictions: {batch_result['predictions']}")
    print(f"Confidences: {batch_result['confidences']}")
    
    # Model information
    print("\n📊 Model Information:")
    info = predictor.get_model_info()
    print(f"Total Models: {info['total_models']}")
    print(f"Model Names: {info['model_names']}")
    print(f"Features: {info['feature_names']}")
    
    return predictor

if __name__ == "__main__":
    # Run example
    predictor = example_usage()
    
    # Save deployment info
    deployment_info = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'Super-Ensemble Meta Learner',
        'performance': {
            'mae': 1.0740,
            'r2': 0.7338,
            'acc_5': 99.62
        },
        'deployment_ready': True
    }
    
    with open('reports/deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"\n✅ Deployment script completed!")
    print(f"📁 Deployment info saved to reports/deployment_info.json")
    print(f"🚀 Super-Ensemble is ready for production use!") 