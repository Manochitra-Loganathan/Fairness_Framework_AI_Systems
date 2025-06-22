# Fairness_Framework_AI_Systems
This codebase includes data preprocessing, balancing (SMOTE, GANs), fairness metrics (SPD, DI, AOD, EOD), model training, mitigation steps, and SHAP-based interpretability.

data/ - Includes both raw and processed versions of all five datasets:
  •	Cleaned and grouped data
  •	Unencoded and encoded formats
  •	Label mappings and group definitions for fairness analysis
  
src/preprocessing/ - Contains all scripts used to prepare data for training and evaluation:
  • Cleaning and formatting scripts
  • Encoding protected attributes
  • Grouping for fairness analysis
  
src/models/ - Includes scripts for training and evaluating models:
  • Random Forest, XGBoost, LightGBM, and TabNet models
  • SHAP-based feature importance analysis
  • Threshold tuning and fairness metric calculations

us_adult_census/, us_diabetes/, nz_census/, nz_acc/, taiwan_credit/ - Each dataset has its own folder with results and code organised by stage:
• base_model/: Baseline model performance and fairness evaluation
Mitigation techniques applied only for datasets observed with disparities: U.S. Census and NZ Census
• augmentation/: Data balancing experiments using SMOTE variants and GANs
• mitigation/: Bias mitigation using Reweighing, Adversarial Debiasing (ADB), and Calibrated Equalised Odds (CEO)

outputs/ - Stores generated results and visual summaries for each dataset in separate subfolders:
• Evaluation metrics, comparison tables, and charts
• SHAP summary plots and feature impact heatmaps
• Threshold tuning results and final model outputs
