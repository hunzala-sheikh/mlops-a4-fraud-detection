"""KFP v2 lightweight components for fraud-detection pipeline.

7 components per A4 Task 1:
  1. ingest_data
  2. validate_data
  3. preprocess_data
  4. feature_engineer
  5. train_model     (XGBoost / LightGBM / RF+FS hybrid, optional cost-sensitive)
  6. evaluate_model  (precision/recall/F1/AUC-ROC + confusion matrix)
  7. deploy_model    (conditional - only if AUC-ROC >= threshold)
"""
