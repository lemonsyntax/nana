
# Student Performance Prediction - Model Report

## Executive Summary
- **Model Type**: Neural Network (regression)
- **Training Date**: 2025-07-10 01:19:26
- **Dataset Size**: 6606 samples
- **Features**: 13 engineered features

## Model Performance
- **MSE**: 3.5007
- **MAE**: 1.0740
- **RMSE**: 1.8710
- **R2**: 0.7353
- **MAPE**: 1.5776

## Model Architecture
- **Hidden Layers**: [64, 32, 16]
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.001
- **Activation**: relu

## Top 10 Most Important Features
1. **Attendance**: 0.1449
2. **Hours_Studied**: 0.1294
3. **Previous_Scores**: 0.0720
4. **Tutoring_Sessions**: 0.0708
5. **Family_Income**: 0.0702
6. **Teacher_Quality**: 0.0689
7. **Motivation_Level**: 0.0686
8. **Physical_Activity**: 0.0683
9. **Extracurricular_Activities**: 0.0659
10. **Parental_Education_Level**: 0.0653

## Files Generated
- Model: models/student_performance_model_20250710_011926.keras
- Preprocessor: models/preprocessor_20250710_011926.pkl
- Plots: plots/
- Reports: reports/

## Next Steps
1. Deploy the model using the saved files
2. Monitor model performance in production
3. Retrain periodically with new data
4. Consider ensemble methods for improved performance
