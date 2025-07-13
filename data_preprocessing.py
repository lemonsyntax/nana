#!/usr/bin/env python3
"""
Data Preprocessing Module for Student Performance Prediction

This module handles:
1. Data cleaning and validation
2. Categorical variable encoding
3. Feature scaling and normalization
4. Train-test splitting
5. Feature engineering
6. Outlier detection and handling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles comprehensive data preprocessing for student performance prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = []
        self.categorical_columns = []
        self.numeric_columns = []
        self.is_fitted = False
        self.outlier_threshold = 3.0  # Z-score threshold for outlier detection
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = 'Exam_Score', 
                       test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Comprehensive preprocessing pipeline with feature scaling and validation
        """
        print("Starting comprehensive data preprocessing...")
        
        # Data validation
        self._validate_data(df, target_column)
        
        # Select relevant features (expanded from 5 to more features)
        feature_cols = self._select_features(df, target_column)
        df = df.loc[:, feature_cols + [target_column]].copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Detect and handle outliers
        df = self._handle_outliers(df, target_column)
        
        # Encode categorical variables
        df = self._encode_categorical_variables(df)
        
        # Split data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        self.feature_columns = X.columns.tolist()
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Fit scaler on training data and transform both sets
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.is_fitted = True
        print(f"✅ Data preprocessing completed! Using {len(self.feature_columns)} features")
        print(f"   Features: {', '.join(self.feature_columns)}")
        
        return X_train_scaled, X_test_scaled, np.array(y_train), np.array(y_test)
    
    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the same preprocessing steps
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data")
        
        # Select the same features used during training
        df = df.loc[:, self.feature_columns].copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Encode categorical variables (using same mappings)
        df = self._encode_categorical_variables(df)
        
        # Scale features using the fitted scaler
        X_scaled = self.scaler.transform(df)
        
        return X_scaled
    
    def get_feature_names(self) -> List[str]:
        """Get the names of features used in the model"""
        return self.feature_columns.copy()
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing steps"""
        return {
            'is_fitted': self.is_fitted,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numeric_columns': self.numeric_columns,
            'label_encoders': list(self.label_encoders.keys()),
            'outlier_threshold': self.outlier_threshold
        }
    
    def _validate_data(self, df: pd.DataFrame, target_column: str) -> None:
        """Validate data quality and structure"""
        print("   Validating data...")
        
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Check for required columns
        required_cols = ['Hours_Studied', 'Previous_Scores', 'Attendance']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing: {missing_cols}")
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise ValueError(f"Target column '{target_column}' must be numeric")
        
        print(f"   ✅ Data validation passed: {len(df)} samples, {len(df.columns)} features")
    
    def _select_features(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """Select relevant features for the model"""
        # Core academic features
        core_features = [
            'Hours_Studied', 'Previous_Scores', 'Attendance'
        ]
        
        # Additional features if available
        additional_features = [
            'Extracurricular_Activities', 'Parental_Education_Level',
            'Sleep_Hours', 'Tutoring_Sessions',
            'Family_Income', 'Teacher_Quality', 'Peer_Influence',
            'Internet_Access'
        ]
        
        # Select features that exist in the dataset
        available_features = []
        for feature in core_features + additional_features:
            if feature in df.columns and feature != target_column:
                available_features.append(feature)
        
        print(f"   Selected {len(available_features)} features: {', '.join(available_features)}")
        return available_features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"   Handling {missing_counts.sum()} missing values...")
            
            # For numeric columns, use mean imputation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # For categorical columns, use mode imputation
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Detect and handle outliers using Z-score method"""
        print("   Checking for outliers...")
        
        # Only check numeric columns (excluding target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_column]
        
        outliers_found = 0
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > self.outlier_threshold
            
            if outliers.sum() > 0:
                outliers_found += outliers.sum()
                # Cap outliers at 3 standard deviations
                df.loc[outliers, col] = df[col].mean() + self.outlier_threshold * df[col].std()
        
        if outliers_found > 0:
            print(f"   ✅ Handled {outliers_found} outliers using capping method")
        else:
            print("   ✅ No significant outliers found")
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables to numeric format"""
        print("   Encoding categorical variables...")
        
        # Define encoding mappings
        encodings = {
            'Extracurricular_Activities': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0},
            'Parental_Education_Level': {
                'High School': 1, 'College': 2, 'Undergraduate': 2, 
                'Graduate': 3, 'Postgraduate': 4, 1: 1, 2: 2, 3: 3, 4: 4
            },
            'Family_Income': {'Low': 1, 'Medium': 2, 'High': 3, 'low': 1, 'medium': 2, 'high': 3},
            'Teacher_Quality': {'Low': 1, 'Medium': 2, 'High': 3, 'low': 1, 'medium': 2, 'high': 3},
            'Peer_Influence': {'Negative': 1, 'Neutral': 2, 'Positive': 3, 'negative': 1, 'neutral': 2, 'positive': 3},
            'Internet_Access': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
        }
        
        # Apply encodings
        for col, mapping in encodings.items():
            if col in df.columns:
                df[col] = df[col].replace(mapping)
                # Convert to numeric, handling any remaining non-numeric values
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        print("   ✅ Categorical encoding completed")
        return df

def main():
    """Test the data preprocessing module"""
    from data_collection import DataCollector
    
    print("=" * 60)
    print("DATA PREPROCESSING MODULE TEST")
    print("=" * 60)
    
    # Load data
    collector = DataCollector()
    df = collector.load_csv()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)
    
    # Get preprocessing information
    info = preprocessor.get_preprocessing_info()
    
    print(f"\nPreprocessing Information:")
    print(f"Fitted: {info['is_fitted']}")
    print(f"Total features: {len(info['feature_columns'])}")
    print(f"Outlier threshold: {info['outlier_threshold']}")
    print(f"Label encoded features: {len(info['label_encoders'])}")
    
    print(f"\nFeature names (first 10):")
    for i, feature in enumerate(info['feature_columns'][:10]):
        print(f"  {i+1}. {feature}")
    
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Test transform_new_data
    print(f"\nTesting transform_new_data...")
    sample_data = df.head(3).copy()
    transformed_sample = preprocessor.transform_new_data(sample_data)
    print(f"Sample transformation shape: {transformed_sample.shape}")
    
    print(f"\n✅ Enhanced data preprocessing completed successfully!")

if __name__ == "__main__":
    main() 