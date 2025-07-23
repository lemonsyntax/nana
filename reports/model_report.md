
# Student Performance Prediction - Model Report

## Executive Summary
- **Model Type**: Neural Network (classification)
- **Training Date**: 2025-07-15 12:45:42
- **Dataset Size**: 6606 samples
- **Features**: 11 engineered features

## Model Performance
- **ACCURACY**: 1.0000
- **PRECISION**: 1.0000
- **RECALL**: 1.0000
- **F1**: 1.0000

## Model Architecture
- **Hidden Layers**: [64, 32, 16]
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.001
- **Activation**: relu

## Top 10 Most Important Features
1. **Previous_Scores**: 0.1123
2. **Sleep_Hours**: 0.1063
3. **Peer_Influence**: 0.1047
4. **Extracurricular_Activities**: 0.1027
5. **Attendance**: 0.1015
6. **Tutoring_Sessions**: 0.0956
7. **Family_Income**: 0.0934
8. **Teacher_Quality**: 0.0890
9. **Hours_Studied**: 0.0681
10. **Internet_Access**: 0.0633

## Files Generated
- Model: models/student_performance_model_20250715_124541.keras
- Preprocessor: models/preprocessor_20250715_124542.pkl
- Plots: plots/
- Reports: reports/

## Next Steps
1. Deploy the model using the saved files
2. Monitor model performance in production
3. Retrain periodically with new data
4. Consider ensemble methods for improved performance
