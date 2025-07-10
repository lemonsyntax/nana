
# Student Performance Prediction - Model Report

## Executive Summary
- **Model Type**: Neural Network (regression)
- **Training Date**: 2025-07-10 02:06:50
- **Dataset Size**: 6606 samples
- **Features**: 13 engineered features

## Model Performance
- **MSE**: 3.9962
- **MAE**: 1.2047
- **RMSE**: 1.9990
- **R2**: 0.6978
- **MAPE**: 1.7644
- **GRADE_ACCURACY_5**: 99.6218
- **GRADE_ACCURACY_10**: 99.6218
- **CORRELATION**: 0.8539

## Model Architecture
- **Hidden Layers**: [64, 32, 16]
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.001
- **Activation**: relu

## Top 10 Most Important Features
1. **Attendance**: 0.2331
2. **Hours_Studied**: 0.1778
3. **Previous_Scores**: 0.0884
4. **Tutoring_Sessions**: 0.0669
5. **Family_Income**: 0.0520
6. **Teacher_Quality**: 0.0517
7. **Parental_Education_Level**: 0.0513
8. **Physical_Activity**: 0.0511
9. **Peer_Influence**: 0.0503
10. **Sleep_Hours**: 0.0469

## Files Generated
- Model: models/student_performance_model_20250710_020650.keras
- Preprocessor: models/preprocessor_20250710_020650.pkl
- Plots: plots/
- Reports: reports/

## Next Steps
1. Deploy the model using the saved files
2. Monitor model performance in production
3. Retrain periodically with new data
4. Consider ensemble methods for improved performance
