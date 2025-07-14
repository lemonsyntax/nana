
# Student Performance Prediction - Model Report

## Executive Summary
- **Model Type**: Neural Network (regression)
- **Training Date**: 2025-07-14 05:02:31
- **Dataset Size**: 6606 samples
- **Features**: 11 engineered features

## Model Performance
- **MSE**: 4.4178
- **MAE**: 1.2968
- **RMSE**: 2.1019
- **R2**: 0.6659
- **MAPE**: 1.8889
- **GRADE_ACCURACY_5**: 99.5461
- **GRADE_ACCURACY_10**: 99.6218
- **CORRELATION**: 0.8488

## Model Architecture
- **Hidden Layers**: [64, 32, 16]
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.001
- **Activation**: relu

## Top 10 Most Important Features
1. **Attendance**: 0.2127
2. **Hours_Studied**: 0.1515
3. **Previous_Scores**: 0.0964
4. **Tutoring_Sessions**: 0.0865
5. **Internet_Access**: 0.0758
6. **Family_Income**: 0.0701
7. **Teacher_Quality**: 0.0688
8. **Parental_Education_Level**: 0.0672
9. **Extracurricular_Activities**: 0.0609
10. **Peer_Influence**: 0.0608

## Files Generated
- Model: models/student_performance_model_20250714_050231.keras
- Preprocessor: models/preprocessor_20250714_050231.pkl
- Plots: plots/
- Reports: reports/

## Next Steps
1. Deploy the model using the saved files
2. Monitor model performance in production
3. Retrain periodically with new data
4. Consider ensemble methods for improved performance
