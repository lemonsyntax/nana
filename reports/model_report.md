
# Student Performance Prediction - Model Report

## Executive Summary
- **Model Type**: Neural Network (regression)
- **Training Date**: 2025-07-12 11:02:26
- **Dataset Size**: 6606 samples
- **Features**: 11 engineered features

## Model Performance
- **MSE**: 13.2243
- **MAE**: 2.7439
- **RMSE**: 3.6365
- **R2**: -0.0000
- **MAPE**: 4.0790
- **GRADE_ACCURACY_5**: 86.2330
- **GRADE_ACCURACY_10**: 99.3192
- **CORRELATION**: nan

## Model Architecture
- **Hidden Layers**: [64, 32, 16]
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.001
- **Activation**: relu

## Top 10 Most Important Features
1. **Family_Income**: 0.1407
2. **Teacher_Quality**: 0.1360
3. **Parental_Education_Level**: 0.1075
4. **Previous_Scores**: 0.1053
5. **Hours_Studied**: 0.0973
6. **Extracurricular_Activities**: 0.0962
7. **Internet_Access**: 0.0948
8. **Sleep_Hours**: 0.0755
9. **Attendance**: 0.0660
10. **Tutoring_Sessions**: 0.0536

## Files Generated
- Model: models/student_performance_model_20250712_110226.keras
- Preprocessor: models/preprocessor_20250712_110226.pkl
- Plots: plots/
- Reports: reports/

## Next Steps
1. Deploy the model using the saved files
2. Monitor model performance in production
3. Retrain periodically with new data
4. Consider ensemble methods for improved performance
