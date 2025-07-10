
# Student Performance Prediction - Model Report

## Executive Summary
- **Model Type**: Neural Network (regression)
- **Training Date**: 2025-07-10 00:36:56
- **Dataset Size**: 6606 samples
- **Features**: 5 engineered features

## Model Performance
- **MSE**: 7.6229
- **MAE**: 1.9144
- **RMSE**: 2.7610
- **R2**: 0.4236
- **MAPE**: 2.8022

## Model Architecture
- **Hidden Layers**: [64, 32, 16]
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.001
- **Activation**: relu

## Top 10 Most Important Features
1. **Parental_Education_Level**: 0.2382
2. **Hours_Studied**: 0.2281
3. **Extracurricular_Activities**: 0.1910
4. **Attendance**: 0.1832
5. **Previous_Scores**: 0.1596

## Files Generated
- Model: models/student_performance_model_20250710_003656.keras
- Preprocessor: models/preprocessor_20250710_003656.pkl
- Plots: plots/
- Reports: reports/

## Next Steps
1. Deploy the model using the saved files
2. Monitor model performance in production
3. Retrain periodically with new data
4. Consider ensemble methods for improved performance
