# Project Documentation

## Overview
This project implements a comprehensive neural network-based system for predicting student performance using advanced machine learning techniques. The system includes data collection, preprocessing, model training, evaluation, and a web interface for predictions. **The current system is designed to use only a single dataset: `data/StudentPerformanceFactors.csv`.**

---

## Motivation and Problem Statement
Accurate prediction of student performance is challenging due to the complex and multifaceted nature of the factors influencing academic outcomes. Traditional machine learning algorithms and models often fall short in capturing these complexities. This project addresses these issues by implementing a complete neural network-based system that includes:
- Data collection from a single, comprehensive CSV file
- Comprehensive preprocessing with feature engineering
- Configurable neural network architectures with regularization
- Automated hyperparameter optimization
- Model interpretability through feature importance analysis
- Web-based interface for practical deployment

### Project Objectives

#### Global (General) Objectives
The main goal of this project is to develop and implement a comprehensive predictive system for assessing student performance using neural networks. This involves creating an advanced system that leverages neural network architectures to analyze and forecast academic outcomes based on a single, well-structured dataset, while providing a complete pipeline from data collection to user interface.

#### Specific Objectives
1. **Data Collection and Integration**: Develop a robust data collection system that loads and validates a single CSV dataset (`data/StudentPerformanceFactors.csv`).
2. **Preprocessing Pipeline**: Implement comprehensive data cleaning, feature engineering, and encoding for the dataset.
3. **Neural Network Engine**: Build a configurable neural network for regression tasks.
4. **Training and Optimization**: Automate model training and hyperparameter tuning.
5. **Evaluation System**: Provide comprehensive performance assessment and reporting.
6. **Web Interface**: Deliver a user-friendly application for predictions and insights.
7. **Model Management**: Enable model persistence and versioning.

---

## System Architecture Overview
The system follows a modular design pattern with the following components:
- **Data Collection Module**: Loads and validates the single CSV dataset
- **Preprocessing Pipeline**: Comprehensive data cleaning and feature engineering
- **Neural Network Engine**: Configurable deep learning models
- **Training and Optimization**: Automated hyperparameter tuning
- **Evaluation System**: Comprehensive performance assessment
- **Web Interface**: User-friendly application for predictions
- **Model Management**: Persistence and versioning capabilities

### Data Collection
- The system is now designed to load only `data/StudentPerformanceFactors.csv`.
- No support for UCI datasets, synthetic data, or dynamic data sources in the current version.

### Preprocessing Pipeline
- Handles missing values, feature engineering, encoding, and scaling for the single dataset.

### Neural Network Engine
- Configurable architecture with support for regularization and advanced optimization.

### Training and Evaluation
- Automated training pipeline with early stopping and learning rate scheduling.
- Comprehensive evaluation metrics (MAE, MSE, RMSE, R²).

### Web Interface
- Streamlit-based application for predictions and data insights.

---

## Experimental Setup

### Dataset Configuration
- **Primary Dataset**: `data/StudentPerformanceFactors.csv` (the only dataset used in the current system)

### Model Configurations
- Various neural network configurations can be tested (e.g., number of layers, dropout rate, learning rate).

### Training Parameters
- Optimization: Adam optimizer with learning rate scheduling
- Batch Size: 32 samples per batch for memory efficiency

---

## Updated Project Structure
```
nana/
├── app.py                          # Streamlit web application
├── data_collection.py              # Data collection and loading module
├── data_preprocessing.py           # Data preprocessing and feature engineering
├── neural_network_model.py         # Neural network model implementation
├── train_model.py                  # Main training script
├── test_system.py                  # System testing script
├── requirements.txt                # Python dependencies
├── data/                           # Data directory
│   └── StudentPerformanceFactors.csv
├── models/                         # Trained model files
├── plots/                          # Generated visualizations
└── reports/                        # Analysis reports
```

## Key Features

### 1. Data Collection (`data_collection.py`)
- Loads and validates a single CSV dataset
- Data validation and quality assurance procedures

### 2. Data Preprocessing (`data_preprocessing.py`)
- Comprehensive data cleaning procedures
- Advanced feature engineering techniques
- Automated handling of missing values and outliers
- Feature scaling and encoding implementations

### 3. Neural Network Engine (`neural_network_model.py`)
- Configurable neural network architectures
- Multiple activation function support
- Regularization techniques (dropout, L2)
- Model training and evaluation capabilities

### 4. Training System (`train_model.py`)
- Automated training pipeline
- Early stopping and learning rate scheduling
- Model persistence and versioning

### 5. Web Interface (`app.py`)
- Multi-page Streamlit application
- Interactive prediction forms
- Real-time visualization dashboard
- Comprehensive analysis reports

---

## Methodology
- The system is now focused on a single, high-quality dataset for all experiments and deployment.
- All code, documentation, and the app are now focused on this single dataset.
- If you want to update the data, replace the contents of `data/StudentPerformanceFactors.csv`.

---

## Contact
For questions or contributions, please open an issue or pull request. 