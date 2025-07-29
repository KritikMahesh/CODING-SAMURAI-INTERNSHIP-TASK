
# 📉 Customer Churn Prediction

This project uses various machine learning techniques to predict customer churn based on a given dataset. Built and executed in Google Colab, it demonstrates core steps in a predictive ML pipeline: data preprocessing, visualization, model training, and evaluation.

---

## 📚 Overview

- **Dataset**: Customer churn data (CSV format with demographic, account, and service information)
- **Models**: Logistic Regression, Random Forest, and optionally XGBoost, Decision Tree
- **Goal**: Predict whether a customer will churn (leave) the service
- **Platform**: Google Colab

---

## 🛠️ Tools & Technologies

- Python 3
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🔍 Features Covered

- Data Cleaning: Null values handling, feature dropping
- Categorical Encoding: One-Hot Encoding, Label Encoding
- Exploratory Data Analysis (EDA): Count plots, distribution analysis, correlation heatmap
- Model Training: Logistic Regression, Random Forest, optional advanced models
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Feature Importance: Understanding key churn predictors
- Live Prediction: Test custom input with `predict_churn()` function

---

## 📈 Results

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | ~80%     | ~79%      | ~76%   | ~77%     |
| Random Forest        | ~83%     | ~81%      | ~80%   | ~80%     |

(Note: Results may vary slightly based on data split and random state)

---

## 🚀 How to Run the Project

1. Clone this repository or upload to your GitHub.
2. Open the notebook (`Customer_Churn_Prediction.ipynb`) in **Google Colab**.
3. Run all cells to:
   - Load and preprocess data
   - Perform EDA
   - Train models and evaluate them
   - Run prediction on custom user input

---

## 💡 Live Prediction Example

```python
predict_churn({
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'tenure': 12,
    'MonthlyCharges': 75.5,
    ...
})
# Output: Likely to Churn
```

---

## 🔧 Requirements

Install necessary packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 👤 Author

Kritik Mahesh – [Portfolio](https://kritikmahesh.framer.website)
