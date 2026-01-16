# Twitter Sentiment Analysis (Sentiment140)

## 📌 Project Overview
This project focuses on **sentiment analysis** of Twitter messages using the **Sentiment140 dataset**.  
The goal is to classify tweets into **positive** or **negative** sentiments using classical Machine Learning and Natural Language Processing (NLP) techniques.

This project demonstrates how to handle large-scale textual data and build robust text classification pipelines.

---

## 🎯 Objectives
- Preprocess and clean raw tweet text
- Convert text data into numerical features
- Train and evaluate sentiment classification models
- Compare different machine learning algorithms
- Apply best practices in NLP and Machine Learning

---

## 📂 Dataset
- **Name**: Sentiment140
- **Source**: Kaggle
- **Size**: 1.6 million tweets
- **Classes**:
  - `0` → Negative sentiment
  - `4` → Positive sentiment

---

## 🧰 Tools & Technologies
- Python
- Google Colab
- Pandas & NumPy
- Scikit-learn
- Regular Expressions (`re`)

---

## 🧠 Concepts & Methods Used

### 🔹 Data Preprocessing
- Handling large CSV files
- Encoding handling (`latin-1`)
- Column selection and renaming
- Label encoding (Negative / Positive)
- Text normalization (lowercasing)
- Removing URLs, mentions, and special characters using regex

### 🔹 Feature Engineering
- TF-IDF Vectorization
- N-grams (unigrams and bigrams)
- Dimensionality reduction through feature filtering

### 🔹 Machine Learning Models
- Logistic Regression
- Multinomial Naive Bayes (baseline comparison)

### 🔹 ML Pipeline
- Scikit-learn Pipeline (TF-IDF + Classifier)
- Prevention of data leakage
- Clean and reproducible workflow

### 🔹 Model Evaluation
- Train/Test split
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Classification Report

---

## 📊 Results
- **Accuracy**: ~82%
- Balanced performance across positive and negative classes
- No significant class bias observed
- Naive Bayes used as a baseline, Logistic Regression selected as the main model

---

## 🧪 Example Prediction
Tweet: "I really love this product!"
Prediction: Positive



---

## 🚀 Key Takeaways
- Text preprocessing has a major impact on model performance
- TF-IDF combined with Logistic Regression provides strong baseline results
- Naive Bayes is a fast and effective baseline for text classification
- Pipelines are essential for clean and professional ML projects

---

## 📈 Possible Improvements
- Hyperparameter optimization using GridSearchCV
- Use of trigrams
- Deep Learning models (LSTM, BERT)
- Advanced NLP preprocessing with spaCy or NLTK

---

## 👤 Author
Ahmed Baccari  
Computer Engineering Student – 2nd Year