# Student Performance Prediction

**Note:** This project is now simplified to use only a single dataset: `data/StudentPerformanceFactors.csv`. All data loading and processing is hardcoded to this file for clarity and maintainability.

## Overview
This project predicts student performance using a neural network trained on a single, comprehensive dataset. The pipeline includes data collection, preprocessing, model training, evaluation, and reporting.

## Dataset
- The only dataset used is: `data/StudentPerformanceFactors.csv`
- All scripts and the app are configured to use this file exclusively.
- **Features**: 11 engineered features including study habits, academic history, family background, and support systems
- **Target**: Exam_Score (0-100 scale)

## Usage
1. **Train the Model:**
   ```bash
   python train_model.py
   ```
   - Outputs will be saved in the `models/`, `plots/`, and `reports/` directories.

2. **Run the Web App:**
   ```bash
   streamlit run app.py
   ```
   - The app will use the same dataset for predictions and analysis.

## Directory Structure
- `data/StudentPerformanceFactors.csv` — The only dataset used
- `models/` — Saved models and preprocessors
- `plots/` — Visualizations
- `reports/` — Model reports and summaries

## Notes
- All code, documentation, and the app are now focused on this single dataset.
- If you want to update the data, replace the contents of `data/StudentPerformanceFactors.csv`.

## Requirements
See `requirements.txt` for dependencies.

## Contact
For questions or contributions, please open an issue or pull request. 