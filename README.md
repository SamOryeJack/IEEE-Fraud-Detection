# IEEE-CIS Fraud Detection

🏆 **0.93 Private AUC | 0.91 Public AUC** | Kaggle Competition

A machine learning solution for detecting fraudulent online transactions using LightGBM, achieving top-tier performance through advanced feature engineering, hyperparameter optimization, and model interpretability techniques.

---

## 🎯 Overview

This project was developed as part of Maryville University's DSCI 598 Capstone course, tackling the [IEEE-CIS Fraud Detection Challenge](https://www.kaggle.com/c/ieee-fraud-detection) on Kaggle. The goal was to build a binary classification model to identify fraudulent transactions from a dataset of over 590,000 training observations and 432 features.

Our team delivered a high-performing, interpretable solution that balanced accuracy with computational efficiency, demonstrating strong collaboration and technical problem-solving skills.

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **ML Framework:** LightGBM (gradient boosting)
- **Optimization:** Hyperopt (Bayesian hyperparameter tuning)
- **Interpretability:** SHAP (feature importance analysis)
- **Data Processing:** pandas, NumPy
- **Validation:** Stratified K-Fold Cross-Validation
- **Environment:** Jupyter Notebook

---

## ✨ Key Achievements

### Model Performance
- **Private Leaderboard:** 0.93 AUC
- **Public Leaderboard:** 0.91 AUC
- **Evaluation Metric:** Area Under the ROC Curve (AUC)

### Technical Highlights
- **Feature Engineering:** Reduced 432 features to an optimized subset using forward selection
- **Hyperparameter Tuning:** Systematic optimization of learning rate, tree depth, and regularization
- **Model Interpretability:** SHAP analysis to identify key fraud indicators
- **Cross-Validation:** Stratified K-Fold to handle class imbalance and ensure robust performance
- **Early Stopping:** Prevented overfitting while optimizing training time

---

## 🔬 Approach

### 1. Data Preprocessing
- Handled missing values across 432 features (49 categorical, 383 numerical)
- Addressed categorical features with high cardinality (13,000+ levels)
- Merged transaction and identity datasets
- Ensured consistency between training and test sets
- Applied feature scaling and normalization

### 2. Feature Selection
Implemented iterative forward selection to identify the most impactful features:
- Started with empty feature set
- Tested each feature's contribution to AUC score
- Retained only features that improved model performance
- Reduced dimensionality while maintaining predictive power
- Prevented overfitting from redundant features

### 3. Model Training
- **Algorithm:** LightGBM classifier (optimized for large datasets)
- **Split:** 75% training, 25% validation
- **Parameters tuned:**
  - Learning rate
  - Number of estimators
  - Maximum tree depth
  - Subsample fractions
  - Regularization terms
- **Early stopping:** Halted training after 100 rounds without improvement

### 4. Hyperparameter Optimization
- Used Hyperopt for Bayesian optimization
- Explored large parameter search space efficiently
- Balanced model complexity with generalization
- Iteratively refined through 20-30 model versions

### 5. Model Interpretability
- Applied SHAP values to understand feature importance
- Created correlation heatmaps for feature relationships
- Visualized how features influenced predictions
- Generated actionable insights from model decisions

---

## 📊 Results

| Model Version | Private AUC | Public AUC | Key Technique |
|--------------|-------------|------------|---------------|
| Baseline (V1) | 0.85 | 0.885 | Initial LightGBM with basic preprocessing |
| Optimized (V5) | **0.93** | **0.91** | Forward selection + Hyperopt + SHAP |

**Performance Improvements:**
- 8-point improvement in Private AUC (0.85 → 0.93)
- Consistent performance across public/private leaderboards
- Robust model that generalized well to unseen data

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required libraries:
  ```bash
  pip install lightgbm pandas numpy scikit-learn hyperopt shap matplotlib seaborn
  ```

### Installation

1. Clone the repository
```bash
git clone https://github.com/SamOryeJack/IEEE-Fraud-Detection.git
cd IEEE-Fraud-Detection
```

2. Download the competition data
- Visit [Kaggle Competition Page](https://www.kaggle.com/c/ieee-fraud-detection/data)
- Download `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`
- Place files in the `data/` directory

3. Open the notebook
```bash
jupyter notebook
```

4. Run the analysis
- Navigate to `LightGBM_w_HyperOpt_Pipeline/` directory
- Open the latest version notebook
- Execute cells sequentially

---

## 📁 Project Structure

```
IEEE-Fraud-Detection/
├── data/                          # Competition datasets (not included)
├── LightGBM_w_HyperOpt_Pipeline/  # Main model notebooks
│   └── V5_Optimized_Model.ipynb  # Best performing model
├── exploratory_analysis/          # EDA notebooks
└── README.md                      # This file
```

---

## 🔑 Key Learnings

### Technical Challenges
- **High-cardinality features:** Categorical variables with 13,000+ levels required careful encoding strategies
- **Memory constraints:** 2.7GB training dataset demanded efficient preprocessing
- **Feature selection:** Automated forward selection reduced manual effort and improved results
- **Class imbalance:** Stratified cross-validation ensured proper representation of fraud cases
- **Computational resources:** Early stopping and efficient hyperparameter search optimized training time

### Solutions Implemented
- Iterative feature testing to identify optimal subset
- Bayesian optimization (Hyperopt) for efficient hyperparameter search
- Stratified K-Fold cross-validation for reliable performance estimates
- SHAP analysis for model transparency and trust
- Incremental development through 20-30 model iterations

---

## 👥 Team Collaboration

This project was completed as a team capstone, demonstrating:

- **Hybrid collaboration model:** Balanced independent work with team coordination across different time zones
- **Code reviews:** Regular peer reviews identified bugs and improved code quality
- **Iterative development:** Multiple model versions tested different hypotheses and techniques
- **Knowledge sharing:** Team members contributed research, code development, and optimization strategies
- **Integration:** Standalone scripts merged into cohesive final submission

**Collaboration Skills Demonstrated:**
- Asynchronous teamwork across schedules
- Git-based version control
- Code documentation and knowledge transfer
- Constructive feedback and peer review
- Project management and deadline coordination

---

## 🔮 Future Enhancements

- [ ] Experiment with ensemble methods (stacking, blending)
- [ ] Implement neural network approaches (tabular transformers)
- [ ] Add real-time inference pipeline
- [ ] Create interactive dashboard for fraud detection insights
- [ ] Explore additional feature engineering techniques
- [ ] Deploy model as REST API

---

## 📖 Related Links

- [Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Original Team Repository](https://github.com/gregofkickapoo/dsci_598_capstone)

---

## 📄 License

Apache License 2.0

---

## 👤 Author

**Paul Desmond Jack**

- GitHub: [@SamOryeJack](https://github.com/SamOryeJack)
- LinkedIn: [linkedin.com/in/paul-desmond-155495219](https://www.linkedin.com/in/paul-desmond-155495219/)

---

*Developed as part of Maryville University DSCI 598 Capstone Project*
