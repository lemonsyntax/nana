#!/usr/bin/env python3
"""
Data Collection Module for Student Performance Prediction

This module handles:
1. Student Performance CSV data loading
2. Data validation and basic cleaning
3. Data source management
"""

import pandas as pd
import numpy as np
import requests
import os
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    """Handles data collection from the Student Performance CSV file"""
    
    def __init__(self):
        pass
    
    def load_csv(self) -> pd.DataFrame:
        """
        Load the Student Performance CSV file
        Returns:
            DataFrame with student performance data
        """
        filepath = 'data/StudentPerformanceFactors.csv'
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        print(f"Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
        return self._clean_data(df)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Original dataset shape: {df.shape}")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Missing values found:\n{missing_values[missing_values > 0]}")
            df = df.fillna(df.mode().iloc[0])
        if 'Exam_Score' not in df.columns:
            raise ValueError("No valid target column found. Expected 'Exam_Score'.")
        # Ensure numeric columns are properly typed
        numeric_columns = ['Exam_Score']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Remove rows with invalid Exam_Score values
        df = df[df['Exam_Score'].between(0, 100)]
        print(f"Cleaned dataset shape: {df.shape}")
        return df
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'target_column': 'Exam_Score'
        }
        if 'Exam_Score' in df.columns:
            target_stats = df['Exam_Score'].describe()
            info['target_statistics'] = target_stats.to_dict()
        return info
    
    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to: {filepath}")

def main():
    """Test the data collection module"""
    collector = DataCollector()
    print("=" * 60)
    print("DATA COLLECTION MODULE TEST")
    print("=" * 60)
    df = collector.load_csv()
    info = collector.get_data_info(df)
    print(f"\nDataset Information:")
    print(f"Shape: {info['shape']}")
    print(f"Target column: {info['target_column']}")
    print(f"Numeric columns: {len(info['numeric_columns'])}")
    print(f"Categorical columns: {len(info['categorical_columns'])}")
    if info['target_column']:
        print(f"\nTarget variable statistics:")
        for stat, value in info['target_statistics'].items():
            print(f"  {stat}: {value:.2f}")
    collector.save_dataset(df, 'student_performance_data.csv')
    print(f"\n✅ Data collection completed successfully!")
    print(f"Dataset saved as: student_performance_data.csv")

if __name__ == "__main__":
    main() 