#!/usr/bin/env python3
"""
Comprehensive Accuracy Analysis for Student Performance Prediction Model

This script provides detailed analysis of model prediction accuracy including:
1. Overall performance metrics
2. Accuracy by score ranges
3. Error distribution analysis
4. Feature importance impact on accuracy
5. Confidence intervals
6. Practical accuracy assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_collection import DataCollector
from data_preprocessing import DataPreprocessor
from neural_network_model import StudentPerformanceModel

class AccuracyAnalyzer:
    """Comprehensive accuracy analysis for the student performance prediction model"""
    
    def __init__(self):
        self.collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.results = {}
        
    def load_model_and_data(self):
        """Load the trained model and test data"""
        print("Loading model and data...")
        
        # Load the latest model
        models_dir = 'models'
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(models_dir, latest_model)
        
        # Load model
        self.model = StudentPerformanceModel(input_dim=13)
        self.model.load_model(model_path)
        
        # Load and preprocess data
        df = self.collector.load_csv()
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_data(df)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        self.results = {
            'y_true': y_test,
            'y_pred': y_pred.flatten(),
            'feature_names': self.preprocessor.get_feature_names()
        }
        
        print(f"✅ Model loaded: {latest_model}")
        print(f"✅ Test data: {len(y_test)} samples")
        
    def calculate_comprehensive_metrics(self):
        """Calculate comprehensive accuracy metrics"""
        y_true = self.results['y_true']
        y_pred = self.results['y_pred']
        
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Percentage metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Grade accuracy metrics
        grade_accuracy_1 = np.mean(np.abs(y_true - y_pred) <= 1) * 100
        grade_accuracy_2 = np.mean(np.abs(y_true - y_pred) <= 2) * 100
        grade_accuracy_3 = np.mean(np.abs(y_true - y_pred) <= 3) * 100
        grade_accuracy_5 = np.mean(np.abs(y_true - y_pred) <= 5) * 100
        grade_accuracy_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
        
        # Correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Error distribution
        errors = y_true - y_pred
        error_std = np.std(errors)
        error_median = np.median(np.abs(errors))
        
        # Confidence intervals
        sorted_errors = np.sort(np.abs(errors))
        confidence_90 = np.percentile(sorted_errors, 90)
        confidence_95 = np.percentile(sorted_errors, 95)
        confidence_99 = np.percentile(sorted_errors, 99)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'grade_accuracy_1': grade_accuracy_1,
            'grade_accuracy_2': grade_accuracy_2,
            'grade_accuracy_3': grade_accuracy_3,
            'grade_accuracy_5': grade_accuracy_5,
            'grade_accuracy_10': grade_accuracy_10,
            'correlation': correlation,
            'error_std': error_std,
            'error_median': error_median,
            'confidence_90': confidence_90,
            'confidence_95': confidence_95,
            'confidence_99': confidence_99
        }
        
        self.results['metrics'] = metrics
        return metrics
    
    def analyze_accuracy_by_score_ranges(self):
        """Analyze accuracy across different score ranges"""
        y_true = self.results['y_true']
        y_pred = self.results['y_pred']
        
        # Define score ranges
        ranges = [
            (0, 60, 'Failing (0-60)'),
            (60, 70, 'D Range (60-70)'),
            (70, 80, 'C Range (70-80)'),
            (80, 90, 'B Range (80-90)'),
            (90, 100, 'A Range (90-100)')
        ]
        
        range_analysis = []
        
        for min_score, max_score, label in ranges:
            mask = (y_true >= min_score) & (y_true < max_score)
            if mask.sum() > 0:
                y_true_range = y_true[mask]
                y_pred_range = y_pred[mask]
                
                mae_range = mean_absolute_error(y_true_range, y_pred_range)
                accuracy_5_range = np.mean(np.abs(y_true_range - y_pred_range) <= 5) * 100
                count = len(y_true_range)
                
                range_analysis.append({
                    'range': label,
                    'count': count,
                    'mae': mae_range,
                    'accuracy_5': accuracy_5_range,
                    'percentage': (count / len(y_true)) * 100
                })
        
        self.results['range_analysis'] = range_analysis
        return range_analysis
    
    def analyze_error_distribution(self):
        """Analyze the distribution of prediction errors"""
        y_true = self.results['y_true']
        y_pred = self.results['y_pred']
        errors = y_true - y_pred
        
        error_stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'positive_errors': np.sum(errors > 0),
            'negative_errors': np.sum(errors < 0),
            'zero_errors': np.sum(errors == 0),
            'overpredictions': np.sum(errors < 0),  # Model predicts higher than actual
            'underpredictions': np.sum(errors > 0)  # Model predicts lower than actual
        }
        
        self.results['error_stats'] = error_stats
        return error_stats
    
    def create_accuracy_visualizations(self):
        """Create comprehensive accuracy visualizations"""
        y_true = self.results['y_true']
        y_pred = self.results['y_pred']
        errors = y_true - y_pred
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Accuracy Analysis - Student Performance Prediction', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Exam Scores')
        axes[0, 0].set_ylabel('Predicted Exam Scores')
        axes[0, 0].set_title('Actual vs Predicted Scores')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² and correlation info
        r2 = self.results['metrics']['r2']
        corr = self.results['metrics']['correlation']
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}\nCorrelation = {corr:.3f}', 
                       transform=axes[0, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Error distribution histogram
        axes[0, 1].hist(errors, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add error statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        axes[0, 1].text(0.05, 0.95, f'Mean Error: {mean_error:.2f}\nStd Error: {std_error:.2f}', 
                       transform=axes[0, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 3. Grade accuracy by tolerance
        tolerances = [1, 2, 3, 5, 10]
        accuracies = [
            self.results['metrics']['grade_accuracy_1'],
            self.results['metrics']['grade_accuracy_2'],
            self.results['metrics']['grade_accuracy_3'],
            self.results['metrics']['grade_accuracy_5'],
            self.results['metrics']['grade_accuracy_10']
        ]
        
        bars = axes[0, 2].bar(tolerances, accuracies, color='orange', alpha=0.7)
        axes[0, 2].set_xlabel('Tolerance (± points)')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].set_title('Grade Accuracy by Tolerance')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Accuracy by score ranges
        range_analysis = self.results['range_analysis']
        ranges = [item['range'] for item in range_analysis]
        accuracies_range = [item['accuracy_5'] for item in range_analysis]
        counts = [item['count'] for item in range_analysis]
        
        bars = axes[1, 0].bar(ranges, accuracies_range, color='purple', alpha=0.7)
        axes[1, 0].set_xlabel('Score Ranges')
        axes[1, 0].set_ylabel('Accuracy within ±5 points (%)')
        axes[1, 0].set_title('Accuracy by Score Ranges')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # 5. Residuals plot
        axes[1, 1].scatter(y_pred, errors, alpha=0.6, color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Predicted Scores')
        axes[1, 1].set_ylabel('Residuals (Actual - Predicted)')
        axes[1, 1].set_title('Residuals Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Cumulative accuracy
        sorted_errors = np.sort(np.abs(errors))
        cumulative_percent = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        
        axes[1, 2].plot(sorted_errors, cumulative_percent, linewidth=2, color='brown')
        axes[1, 2].set_xlabel('Absolute Error')
        axes[1, 2].set_ylabel('Cumulative Percentage (%)')
        axes[1, 2].set_title('Cumulative Error Distribution')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add key points
        for threshold in [1, 2, 3, 5]:
            idx = np.searchsorted(sorted_errors, threshold)
            if idx < len(cumulative_percent):
                axes[1, 2].axvline(x=threshold, color='red', linestyle=':', alpha=0.7)
                axes[1, 2].text(threshold, cumulative_percent[idx], f'{cumulative_percent[idx]:.1f}%', 
                               ha='left', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('plots/comprehensive_accuracy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comprehensive accuracy visualizations saved to plots/comprehensive_accuracy_analysis.png")
    
    def print_detailed_accuracy_report(self):
        """Print a detailed accuracy report"""
        metrics = self.results['metrics']
        range_analysis = self.results['range_analysis']
        error_stats = self.results['error_stats']
        
        print("\n" + "="*80)
        print("COMPREHENSIVE ACCURACY ANALYSIS REPORT")
        print("="*80)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Samples: {len(self.results['y_true'])}")
        
        print("\n📊 OVERALL PERFORMANCE METRICS:")
        print("-" * 50)
        print(f"R² Score (Variance Explained): {metrics['r2']:.4f} ({metrics['r2']*100:.2f}%)")
        print(f"Correlation Coefficient: {metrics['correlation']:.4f}")
        print(f"Mean Absolute Error: {metrics['mae']:.4f} points")
        print(f"Root Mean Square Error: {metrics['rmse']:.4f} points")
        print(f"Mean Absolute Percentage Error: {metrics['mape']:.4f}%")
        
        print("\n🎯 GRADE PREDICTION ACCURACY:")
        print("-" * 50)
        print(f"Within ±1 point: {metrics['grade_accuracy_1']:.2f}%")
        print(f"Within ±2 points: {metrics['grade_accuracy_2']:.2f}%")
        print(f"Within ±3 points: {metrics['grade_accuracy_3']:.2f}%")
        print(f"Within ±5 points: {metrics['grade_accuracy_5']:.2f}%")
        print(f"Within ±10 points: {metrics['grade_accuracy_10']:.2f}%")
        
        print("\n📈 CONFIDENCE INTERVALS:")
        print("-" * 50)
        print(f"90% of predictions within ±{metrics['confidence_90']:.2f} points")
        print(f"95% of predictions within ±{metrics['confidence_95']:.2f} points")
        print(f"99% of predictions within ±{metrics['confidence_99']:.2f} points")
        
        print("\n📊 ACCURACY BY SCORE RANGES:")
        print("-" * 50)
        for item in range_analysis:
            print(f"{item['range']:15} | {item['count']:4} samples | "
                  f"MAE: {item['mae']:.2f} | Accuracy ±5: {item['accuracy_5']:.1f}%")
        
        print("\n🔍 ERROR ANALYSIS:")
        print("-" * 50)
        print(f"Mean Error: {error_stats['mean_error']:.4f} points")
        print(f"Error Standard Deviation: {error_stats['std_error']:.4f} points")
        print(f"Median Absolute Error: {error_stats['median_error']:.4f} points")
        print(f"Overpredictions (Model > Actual): {error_stats['overpredictions']} ({error_stats['overpredictions']/len(self.results['y_true'])*100:.1f}%)")
        print(f"Underpredictions (Model < Actual): {error_stats['underpredictions']} ({error_stats['underpredictions']/len(self.results['y_true'])*100:.1f}%)")
        print(f"Perfect Predictions: {error_stats['zero_errors']} ({error_stats['zero_errors']/len(self.results['y_true'])*100:.1f}%)")
        
        print("\n🏆 PRACTICAL ASSESSMENT:")
        print("-" * 50)
        if metrics['grade_accuracy_5'] >= 95:
            print("✅ EXCELLENT: Model provides highly accurate predictions for practical use")
        elif metrics['grade_accuracy_5'] >= 90:
            print("✅ VERY GOOD: Model provides reliable predictions for most applications")
        elif metrics['grade_accuracy_5'] >= 80:
            print("✅ GOOD: Model provides acceptable predictions with some limitations")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Model accuracy may not be sufficient for practical use")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 50)
        if metrics['grade_accuracy_5'] >= 95:
            print("• Model is ready for production deployment")
            print("• Can be used for high-stakes decisions")
            print("• Consider monitoring for drift over time")
        elif metrics['grade_accuracy_5'] >= 90:
            print("• Model is suitable for most educational applications")
            print("• Use with confidence for general predictions")
            print("• Consider ensemble methods for critical decisions")
        else:
            print("• Consider retraining with more data or different features")
            print("• Review feature engineering and model architecture")
            print("• Test with different hyperparameters")
        
        print("\n" + "="*80)

def main():
    """Run comprehensive accuracy analysis"""
    print("🔍 COMPREHENSIVE ACCURACY ANALYSIS")
    print("="*50)
    
    # Initialize analyzer
    analyzer = AccuracyAnalyzer()
    
    # Load model and data
    analyzer.load_model_and_data()
    
    # Calculate metrics
    print("\n📊 Calculating comprehensive metrics...")
    metrics = analyzer.calculate_comprehensive_metrics()
    
    # Analyze by score ranges
    print("📈 Analyzing accuracy by score ranges...")
    range_analysis = analyzer.analyze_accuracy_by_score_ranges()
    
    # Analyze error distribution
    print("🔍 Analyzing error distribution...")
    error_stats = analyzer.analyze_error_distribution()
    
    # Create visualizations
    print("📊 Creating comprehensive visualizations...")
    analyzer.create_accuracy_visualizations()
    
    # Print detailed report
    analyzer.print_detailed_accuracy_report()
    
    print("\n✅ Comprehensive accuracy analysis completed!")
    print("📁 Check 'plots/comprehensive_accuracy_analysis.png' for visualizations")

if __name__ == "__main__":
    main() 