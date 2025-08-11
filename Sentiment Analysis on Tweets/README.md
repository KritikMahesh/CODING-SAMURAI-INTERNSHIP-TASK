# 💬 Sentiment Analysis on Tweets

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hPQ11qTCSd6k2a7qeQjACY6Z1W5A0flZ)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)

> **Classify tweet sentiment as Positive or Negative using machine learning**

This project implements a Sentiment Analysis model to classify tweets as Positive or Negative. It utilizes a Logistic Regression classifier with TF-IDF vectorization for feature extraction. The model is trained on a preprocessed dataset of tweets and deployed with a simple CLI interface.

---

## 📚 Overview

- **Dataset**: Preprocessed tweet sentiment dataset (positive/negative labels)
- **Model**: Logistic Regression
- **Vectorization**: TF-IDF
- **Goal**: Classify sentiment of tweets with high accuracy
- **Platform**: Google Colab

---

## 🛠️ Tools & Technologies

- **Python 3**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **NLTK**
- **Seaborn**
- **WordCloud**
- **Google Colab**

---

## 📈 Results

| Metric     | Value     |
|------------|-----------|
| Accuracy   | ~87%      |
| F1 Score   | ~0.86     |

---

## 🚀 How to Run the Project

1. **Clone this repository** or upload it to your GitHub
2. **Open the notebook** in **Google Colab**
3. **Run all cells** to:
   - Load and clean the dataset
   - Extract features using TF-IDF
   - Train and evaluate the model
   - Run predictions interactively on custom inputs

---

## 🔧 Requirements

Install all required dependencies using:

```bash
pip install pandas numpy nltk scikit-learn seaborn wordcloud
