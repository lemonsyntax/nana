# Student Performance Prediction System

## Introduction
This project is a comprehensive machine learning system for predicting student academic performance. It leverages advanced data preprocessing, neural network models, hyperparameter tuning, model comparison, and ensemble learning. The system includes a user-friendly Streamlit web app for interactive predictions and analysis.

---

## Features
- **Data Collection & Preprocessing**: Cleans and prepares student data for modeling.
- **Model Training**: Trains neural networks and other models on student data.
- **Hyperparameter Tuning**: Systematically finds the best model settings.
- **Model Comparison**: Compares multiple models to select the best.
- **Super-Ensemble Deployment**: Combines multiple models for robust predictions.
- **Web App**: Interactive Streamlit app for predictions, analysis, and visualization.
- **Reports & Visualization**: Generates plots and JSON reports for insights.

---

## Project Structure
```
├── app.py                        # Streamlit web app
├── data_collection.py            # Data loading and cleaning
├── data_preprocessing.py         # Data preprocessing pipeline
├── train_model.py                # Model training script
├── tune_hyperparameters.py       # Hyperparameter tuning
├── compare_models.py             # Model comparison and analysis
├── accuracy_analysis.py          # Accuracy and performance analysis
├── deploy_super_ensemble.py      # Super-ensemble deployment script
├── models/                       # Trained model files (.keras)
├── data/                         # Raw and processed data files
├── plots/                        # Generated plots and visualizations
├── reports/                      # JSON and markdown reports
├── requirements.txt              # Python dependencies
├── README.md                     # Quick start and summary
└── project_documentation.md      # (This file)
```

---

## Workflow Overview
1. **Data Collection**: Load and clean student data (`data_collection.py`).
2. **Data Preprocessing**: Handle missing values, encode features, scale data, select 11 features (`data_preprocessing.py`).
3. **Model Training**: Train neural networks and save models (`train_model.py`).
4. **Hyperparameter Tuning**: Find best model settings (`tune_hyperparameters.py`).
5. **Model Comparison**: Compare and analyze models (`compare_models.py`, `accuracy_analysis.py`).
6. **Super-Ensemble Creation**: Combine top models for robust predictions (`deploy_super_ensemble.py`).
7. **Web App Deployment**: User-friendly interface for predictions and analysis (`app.py`).
8. **Reports & Visualization**: Generate and review plots and reports (`plots/`, `reports/`).

---

## Setup Instructions
1. **Clone the repository** and navigate to the project folder.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Set up a virtual environment** for isolation.

---

## Usage
### 1. **Data Preparation**
- Place your student data CSV in the `data/` directory (e.g., `StudentPerformanceFactors.csv`).

### 2. **Train a Model**
```bash
python train_model.py
```
- Trains a neural network on the 11 selected features and saves the model in `models/`.

### 3. **Hyperparameter Tuning**
```bash
python tune_hyperparameters.py
```
- Searches for the best model settings and saves the best models.

### 4. **Model Comparison & Analysis**
```bash
python compare_models.py
python accuracy_analysis.py
```
- Compares models and generates performance reports and plots.

### 5. **Super-Ensemble Deployment**
```bash
python deploy_super_ensemble.py
```
- Loads multiple models, combines their predictions, and provides robust ensemble predictions.

### 6. **Run the Web App**
```bash
streamlit run app.py
```
- Opens a browser interface at `http://localhost:8501` for interactive predictions and analysis.

---

## Model Training & Tuning
- **train_model.py**: Trains a neural network using the 11 most relevant features.
- **tune_hyperparameters.py**: Systematically searches for the best hyperparameters (e.g., learning rate, layers, dropout).
- **compare_models.py**: Compares all trained models and outputs performance metrics.
- **accuracy_analysis.py**: Provides detailed accuracy and error analysis.

---

## Ensemble Deployment
- **deploy_super_ensemble.py**: Loads top models, applies ensemble strategies (median, best, stacking), and uses a meta-learner for final predictions. Can be used for batch or API-style predictions.

---

## Reports & Visualization
- **plots/**: Contains PNG plots for model performance, feature importance, and ranking.
- **reports/**: Contains JSON and markdown reports summarizing model results and rankings.

---

## Troubleshooting
- **Files not showing in editor:**
  - Make sure the sidebar is open and the correct folder is selected.
  - Reload the editor window if needed.
- **App not running:**
  - Ensure all dependencies are installed.
  - Check for errors in the terminal when running `streamlit run app.py`.
- **Model input shape errors:**
  - Ensure the app and model both use the same set of 11 features.
- **Data errors:**
  - Check your CSV for missing or misnamed columns.

---

## Future Improvements
- Add more features or data sources for richer predictions.
- Experiment with other ML algorithms (e.g., XGBoost, Random Forest).
- Implement automated retraining as new data arrives.
- Add user authentication and data upload to the web app.
- Deploy the app to a cloud platform for broader access.

---

## Contact & Credits
- Developed by [Your Name/Team].
- For questions or contributions, please open an issue or pull request. 