# 🚀 Sentiment Analysis Using DistilBERT

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green)
![ShadowFox](https://img.shields.io/badge/ShadowFox-AI%2FML%20Internship-orange)

## 📌 Project Overview

This project demonstrates the implementation and evaluation of a Transformer-based Language Model for Sentiment Analysis using DistilBERT.

The goal is to explore how modern Language Models understand textual context, classify sentiment, and outperform traditional machine learning approaches in Natural Language Processing (NLP) tasks.

Developed as part of the **ShadowFox AI/ML Internship Program**, this project focuses on model deployment, performance evaluation, contextual understanding, and visualization of results.

---

# 🎯 Objectives

* Implement a pre-trained Transformer Language Model
* Perform sentiment analysis on movie reviews
* Evaluate model performance using standard metrics
* Analyze contextual understanding capabilities
* Compare Transformer models with traditional ML approaches
* Visualize model predictions and results
* Explore strengths and limitations of modern NLP systems

---

# 🤖 Language Model Used

## DistilBERT

DistilBERT is a lightweight version of BERT that retains most of BERT's language understanding capabilities while being faster and more efficient.

### Model

```python
distilbert-base-uncased-finetuned-sst-2-english
```

### Why DistilBERT?

✔ Faster than BERT

✔ Smaller memory footprint

✔ High sentiment classification accuracy

✔ Easy deployment on standard hardware

✔ State-of-the-art Transformer architecture

---

# 📊 Dataset

## IMDB Movie Reviews Dataset

The project utilizes the IMDB Movie Reviews dataset, which contains labeled reviews categorized as:

| Label | Meaning         |
| ----- | --------------- |
| 0     | Negative Review |
| 1     | Positive Review |

Dataset loaded using:

```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```

---

# 🛠️ Technology Stack

| Technology                | Purpose                       |
| ------------------------- | ----------------------------- |
| Python                    | Programming Language          |
| Jupyter Notebook          | Development Environment       |
| Hugging Face Transformers | Language Model Implementation |
| PyTorch                   | Deep Learning Framework       |
| Datasets                  | Dataset Loading               |
| Scikit-Learn              | Evaluation Metrics            |
| Matplotlib                | Data Visualization            |
| Seaborn                   | Statistical Visualization     |

---

# 📂 Project Structure

```text
ShadowFox/
│
├── Sentiment_Analysis_DistilBERT.ipynb
├── README.md
├── requirements.txt
│
├── images/
│   ├── confusion_matrix.png
│   ├── accuracy_comparison.png
│
└── dataset/
```

---

# 🔄 Project Workflow

```text
IMDB Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
DistilBERT Model
      │
      ▼
Sentiment Prediction
      │
      ▼
Performance Evaluation
      │
      ▼
Visualization
      │
      ▼
Research Analysis
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/shivamkandhway2005-pixel/ShadowFox.git
```

## Navigate to Project Folder

```bash
cd ShadowFox
```

## Install Dependencies

```bash
pip install transformers datasets torch scikit-learn matplotlib seaborn
```

## Launch Jupyter Notebook

```bash
jupyter notebook
```

---

# 📈 Evaluation Metrics

The following metrics are used to assess model performance:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

# 🔍 Research Questions

The project investigates the following questions:

### 1. How accurately can DistilBERT classify sentiment?

### 2. Can DistilBERT understand contextual information?

### 3. How effectively does the model handle negation?

Example:

```text
I liked the movie.
I did not like the movie.
```

### 4. How does the model perform on mixed sentiments?

Example:

```text
The acting was poor but the story was amazing.
```

### 5. Can DistilBERT generalize to different domains?

Examples:

* Healthcare
* Finance
* Technology
* Product Reviews

### 6. How does DistilBERT compare with traditional machine learning models?

---

# 📊 Visualizations

The project includes:

### Confusion Matrix

Measures classification performance.

### Prediction Distribution

Shows positive and negative prediction counts.

### Model Comparison

Comparison between:

* DistilBERT
* Logistic Regression

---

# 🧪 Experimental Analysis

## Context Understanding

The model is tested on:

* Negation
* Ambiguous sentences
* Mixed sentiments
* Context-dependent statements

## Domain Adaptability

Evaluation performed on text from multiple domains:

* Healthcare
* Finance
* Technology
* Food Reviews

## Traditional ML Comparison

Baseline model:

* Logistic Regression + TF-IDF

Compared against:

* DistilBERT Transformer Model

---

# 📌 Key Findings

### Strengths

✔ Strong contextual understanding

✔ High sentiment classification accuracy

✔ Better performance than traditional ML models

✔ Robust handling of common sentiment patterns

### Limitations

✖ Difficulty with sarcasm

✖ Challenges with highly ambiguous statements

✖ Performance may decrease on domain-specific text

✖ Limited understanding of complex reasoning

---

# 🚀 Future Improvements

Potential enhancements include:

* Fine-tuning DistilBERT on IMDB data
* Comparing with BERT and RoBERTa
* Exploring GPT-based models
* Visualizing Transformer attention mechanisms
* Building a Flask/Django web application
* Deploying as a REST API

---

# 📚 Learning Outcomes

This project helped in understanding:

* Transformer Architecture
* Language Models
* Sentiment Analysis
* Hugging Face Ecosystem
* NLP Evaluation Techniques
* Contextual Language Understanding
* Machine Learning Model Comparison
* Data Visualization

---

# 📝 Conclusion

This project successfully demonstrates the deployment and evaluation of a Transformer-based Language Model for sentiment analysis.

The results indicate that DistilBERT provides strong performance while remaining computationally efficient. Through extensive experimentation and analysis, the project highlights the advantages of Transformer architectures in understanding contextual information and generating accurate predictions.

The findings contribute to a deeper understanding of modern NLP systems and their real-world applications.

---

# 🔗 Repository Links

### Project Repository

https://github.com/shivamkandhway2005-pixel/ShadowFox

### GitHub Profile

https://github.com/shivamkandhway2005-pixel

---

# 👨‍💻 Author

**Shivam Kumar**

AI/ML Intern – ShadowFox

GitHub: https://github.com/shivamkandhway2005-pixel

Project Repository: https://github.com/shivamkandhway2005-pixel/ShadowFox

---

# 📄 License

This project is developed for educational and research purposes as part of the ShadowFox AI/ML Internship Program.
