# Predicting Student Performance Using Neural Networks

## Project Overview

This project implements a comprehensive neural network-based system for predicting student performance using advanced machine learning techniques. The system includes data collection, preprocessing, model training, evaluation, and a web interface for predictions.

## Project Structure

```
Phill/
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

## Features

### 1. Data Collection (`data_collection.py`)
- **UCI Dataset Integration**: Loads student performance data from UCI Machine Learning Repository
- **Custom CSV Support**: Handles custom student performance datasets
- **Synthetic Data Generation**: Creates realistic synthetic data when external sources are unavailable
- **Data Validation**: Ensures data quality and completeness

### 2. Data Preprocessing (`data_preprocessing.py`)
- **Data Cleaning**: Handles missing values, duplicates, and outliers
- **Feature Engineering**: Creates new features and transforms existing ones
- **Encoding**: Converts categorical variables to numerical format
- **Scaling**: Normalizes features for optimal neural network performance
- **Data Splitting**: Divides data into training, validation, and test sets

### 3. Neural Network Model (`neural_network_model.py`)
- **Configurable Architecture**: Flexible neural network design with customizable layers
- **Multiple Activation Functions**: ReLU, Sigmoid, Tanh support
- **Optimization Options**: Adam, SGD, RMSprop optimizers
- **Regularization**: Dropout and L2 regularization to prevent overfitting
- **Feature Importance**: SHAP-based feature importance analysis
- **Model Persistence**: Save and load trained models

### 4. Training Pipeline (`train_model.py`)
- **Automated Training**: Complete training pipeline with hyperparameter optimization
- **Cross-Validation**: K-fold cross-validation for robust evaluation
- **Performance Metrics**: Comprehensive evaluation metrics (MAE, MSE, R², RMSE)
- **Visualization**: Training history plots and performance comparisons
- **Model Comparison**: Compares neural network with traditional ML algorithms

### 5. Web Interface (`app.py`)
- **Streamlit Application**: User-friendly web interface
- **Multiple Pages**: Prediction, Analysis, Model Insights, and Reports
- **Interactive Input**: Form-based data input for predictions
- **Real-time Results**: Instant prediction results with confidence scores
- **Visualization**: Interactive charts and performance metrics

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Phill
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the system test:
```bash
python test_system.py
```

## Usage

### Training the Model

1. **Quick Training** (for testing):
```bash
python train_model.py --quick
```

2. **Full Training** (with hyperparameter optimization):
```bash
python train_model.py
```

### Running the Web Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Using the API

The system can also be used programmatically:

```python
from neural_network_model import StudentPerformancePredictor
from data_preprocessing import DataPreprocessor

# Load and preprocess data
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.load_and_preprocess_data()

# Train model
predictor = StudentPerformancePredictor()
predictor.train(X_train, y_train)

# Make predictions
predictions = predictor.predict(X_test)
```

## Model Architecture

The neural network architecture includes:
- **Input Layer**: Adapts to the number of features
- **Hidden Layers**: Configurable dense layers with dropout
- **Output Layer**: Single neuron for regression tasks
- **Activation Functions**: ReLU for hidden layers, linear for output
- **Regularization**: Dropout (0.2) and L2 regularization (0.01)

## Performance Metrics

The system evaluates models using:
- **Mean Absolute Error (MAE)**: Average absolute difference between predictions and actual values
- **Mean Squared Error (MSE)**: Average squared difference
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **R-squared (R²)**: Coefficient of determination
- **Feature Importance**: SHAP-based analysis

## Data Sources

1. **UCI Student Performance Dataset**: Academic performance data with demographic and behavioral features
2. **Custom CSV Files**: User-provided datasets with similar structure
3. **Synthetic Data**: Generated data for testing and development

## Key Features

### Advanced Preprocessing
- Automatic handling of missing values
- Feature scaling and normalization
- Categorical encoding
- Outlier detection and treatment

### Model Optimization
- Grid search for hyperparameter tuning
- Cross-validation for robust evaluation
- Early stopping to prevent overfitting
- Learning rate scheduling

### Comprehensive Evaluation
- Multiple performance metrics
- Visualization of results
- Model comparison with traditional algorithms
- Feature importance analysis

### User-Friendly Interface
- Intuitive web application
- Real-time predictions
- Interactive visualizations
- Detailed analysis reports

## Technical Specifications

### Dependencies
- **Core ML**: TensorFlow, Keras, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Interface**: Streamlit
- **Model Interpretation**: SHAP
- **Additional**: XGBoost, Joblib

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- 2GB disk space for models and data

## Research Contributions

This project addresses several key challenges in student performance prediction:

1. **Data Quality**: Robust preprocessing pipeline for handling real-world educational data
2. **Model Complexity**: Advanced neural network architectures for capturing non-linear relationships
3. **Interpretability**: SHAP-based feature importance for understanding model decisions
4. **Scalability**: Modular design allowing easy extension and modification
5. **Usability**: Web interface for non-technical users

## Future Enhancements

1. **Multi-modal Data**: Integration of text, image, and behavioral data
2. **Real-time Learning**: Online learning capabilities for continuous model updates
3. **Ensemble Methods**: Combination of multiple neural network architectures
4. **Explainable AI**: Enhanced interpretability features
5. **Mobile Application**: Cross-platform mobile interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the student performance dataset
- Streamlit for the web application framework
- TensorFlow and Keras for neural network implementation
- SHAP for model interpretability

## Contact

For questions and support, please contact the development team or create an issue in the repository. 