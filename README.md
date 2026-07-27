# Software Engineering for Machine Learning: AI-Powered Resume Screening and Job Role Prediction System

## Course

**AIMLCZG546 – Software Engineering for Machine Learning**  
Assignment I

---

# Project Overview

This project presents an AI-powered Resume Screening and Job Role Prediction System developed as part of the Software Engineering for Machine Learning (SE4ML) course. The application automatically analyzes resume text and predicts the most suitable job category using Natural Language Processing (NLP) and Machine Learning techniques.

The solution combines machine learning with software engineering principles by applying the GR4ML framework and implementing architectural patterns to build a modular, maintainable, and scalable application.

---

# Problem Statement

Manual resume screening is time-consuming, inconsistent, and difficult to scale when organizations receive a large number of job applications. This project addresses the problem by developing an intelligent resume screening system capable of automatically classifying resumes into predefined job categories.

---

# Dataset

- **Dataset Name:** Resume Dataset
- **File Format:** CSV
- **Total Records:** 2,484
- **Target Variable:** Category
- **Number of Job Categories:** 24

Main columns:

- ID
- Resume_str
- Resume_html
- Category

---

# Technologies Used

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Random Forest Classifier
- Streamlit
- SQLite
- Joblib
- Matplotlib
- Jupyter Notebook

---

# Machine Learning Pipeline

```
Resume Text
      │
      ▼
Text Cleaning
      │
      ▼
TF-IDF Feature Extraction
      │
      ▼
Random Forest Classifier
      │
      ▼
Predicted Job Category
```

---

# Machine Learning Models Evaluated

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Random Forest Classifier
- Multinomial Naïve Bayes

**Selected Model**

Random Forest Classifier

---

# GR4ML Views

The project includes the following GR4ML views:

- Business View
- Analytics Design View
- Data Preparation View

---

# Quality Requirements

The application was designed considering the following quality requirements:

- Performance
- Usability
- Maintainability

---

# System Architecture

The application consists of both Machine Learning (ML) and Non-Machine Learning (Non-ML) components.

## ML Components

- Resume Text Cleaning
- TF-IDF Feature Extraction
- Random Forest Classification
- Job Category Prediction
- Prediction Confidence

## Non-ML Components

- Streamlit User Interface
- SQLite Database
- CQRS Command Module
- CQRS Query Module

---

# Architectural Patterns Implemented

## 1. Pipe-and-Filter

Implemented in the machine learning pipeline:

```
Resume Input
      │
      ▼
Text Cleaning
      │
      ▼
TF-IDF Vectorizer
      │
      ▼
Random Forest Classifier
      │
      ▼
Prediction
      │
      ▼
SQLite Storage
```

## 2. Command Query Responsibility Segregation (CQRS)

The application separates database write and read operations.

### Command Side

- create_database()
- save_prediction()

Implemented in:

```
app/commands.py
```

### Query Side

- view_history()

Implemented in:

```
app/queries.py
```

---

# Application Features

- Resume text input
- Automatic job category prediction
- Prediction confidence score
- SQLite prediction history
- Streamlit-based user interface

---

# Project Structure

```
resume-screening-se4ml/

│
├── app/
│   ├── resume_app.py
│   ├── commands.py
│   └── queries.py
│
├── data/
│   └── Resume.csv
│
├── database/
│   └── prediction_history.db
│
├── models/
│   └── resume_screening_model.pkl
│
├── notebooks/
│   └── Group25.ipynb
│
├── screenshots/
│
├── requirements.txt
│
└── README.md
```

---

# How to Run

## Create Environment

```bash
conda create -n resume-screening python=3.11
conda activate resume-screening
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Launch Application

```bash
streamlit run app/resume_app.py
```

---

# Expected Output

The application allows users to:

- Enter resume text
- Predict the most suitable job category
- Display prediction confidence
- Store prediction history
- View previous predictions

---

# Repository

GitHub Repository:

```
https://github.com/Haridass-K/resume-screening-se4ml
```

---

# Group Details

**Group No:** 25

| Name | Contribution |
|------|--------------|
| Haridass K | Machine learning model development, feature engineering, software architecture implementation, Streamlit application development, and report preparation |
| Sathish T | Requirements analysis, GR4ML Business View, and documentation support |
| Tejaal M | Data preparation, preprocessing, model evaluation, and GR4ML Analytics & Data Preparation Views |
| Sanjayan S | Prototype testing, architecture diagrams, screenshots, and documentation support |

---

# Conclusion

The AI-Powered Resume Screening and Job Role Prediction System successfully demonstrates the integration of Software Engineering for Machine Learning principles with Natural Language Processing and Machine Learning. The application applies the GR4ML framework, implements Pipe-and-Filter and CQRS architectural patterns, and provides a working Streamlit-based prototype for automated resume classification.


---

# Important Submission Notes

## Dataset

The original resume dataset is not included in this compressed submission because of the file size limitations of the Taxila LMS.

If retraining the machine learning model is required, download the dataset from:

**Kaggle Resume Dataset**

https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset

Place the downloaded dataset in:

```
data/Resume.csv
```

---

## Trained Model

The pre-trained Random Forest model is included in this submission:

```
models/resume_screening_model.pkl
```

Therefore, the Streamlit application can be executed directly without retraining the model.

---

## Complete Project

The complete project, including:

- Source code
- Dataset
- Jupyter Notebook
- Architecture diagrams
- Screenshots
- Final report
- Requirements file

is available in the GitHub repository below:

```
https://github.com/Haridass-K/resume-screening-se4ml
```

If any supporting file is unavailable in this compressed submission due to LMS upload size limitations, please refer to the GitHub repository.

---

## Running the Project

1. Create the Python environment.

```bash
conda create -n resume-screening python=3.11
conda activate resume-screening
```

2. Install the required packages.

```bash
pip install -r requirements.txt
```

3. If retraining is required, download the dataset and place it in:

```
data/Resume.csv
```

4. Launch the Streamlit application.

```bash
streamlit run app/resume_app.py
```

Alternatively, open:

```
notebooks/Group25.ipynb
```

and execute all cells to regenerate the machine learning model.

---

## Submission Note

This compressed submission has been prepared to comply with the file upload limitations of the Taxila LMS. The GitHub repository contains the complete project and all supporting files required for evaluation.