
# Student Performance Prediction - Model Report

## Executive Summary
- **Model Type**: Neural Network (regression)
- **Training Date**: 2025-06-27 08:05:41
- **Dataset Size**: 6607 samples
- **Features**: 5 engineered features

## Model Performance
- **MSE**: 10.9161
- **MAE**: 2.3957
- **RMSE**: 3.3040
- **R2**: 0.2277
- **MAPE**: 3.4611

## Model Architecture
- **Hidden Layers**: [64, 32, 16]
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.001
- **Activation**: relu

## Top 10 Most Important Features
1. **Parental_Education_Level**: 0.5177
2. **Extracurricular_Activities**: 0.1311
3. **Hours_Studied**: 0.1276
4. **Attendance**: 0.1213
5. **Previous_Scores**: 0.1023

## Files Generated
- Model: models/student_performance_model_20250627_080541.h5
- Preprocessor: models/preprocessor_20250627_080541.pkl
- Plots: plots/
- Reports: reports/

## Next Steps
1. Deploy the model using the saved files
2. Monitor model performance in production
3. Retrain periodically with new data
4. Consider ensemble methods for improved performance
