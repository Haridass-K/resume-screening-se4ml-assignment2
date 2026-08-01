# Software Engineering for Machine Learning: Production-Ready AI Resume Screening and Job Role Prediction System

## Course

**AIMLCZG546 – Software Engineering for Machine Learning**  
**Assignment II**

---

# Project Overview

This project presents a **Production-Ready AI Resume Screening and Job Role Prediction System** developed as part of the **Software Engineering for Machine Learning (SE4ML)** course.

The application automatically analyzes resume content and predicts the most suitable job category using Natural Language Processing (NLP) and Machine Learning techniques.

Unlike Assignment I, this project extends the research prototype into a production-oriented software system by implementing software engineering best practices including:

- Modular software architecture
- FastAPI REST API
- Streamlit web application
- Automated testing
- Logging
- Security validation
- Model and data quality evaluation
- Production deployment experimentation

---

# Project Components

The project consists of the following major components:

- Research Notebook
- Production Machine Learning Pipeline
- FastAPI REST API
- Streamlit User Interface
- SQLite Prediction Database
- Automated Testing Framework
- Code Quality Validation
- Model & Data Quality Dashboard
- Production Deployment Experimentation

---

# Problem Statement

Manual resume screening is time-consuming, inconsistent, and difficult to scale when organizations receive a large number of job applications.

This project addresses this challenge by developing an intelligent resume screening system capable of automatically classifying resumes into predefined job categories while following production software engineering principles.

---

# Dataset

- **Dataset:** Resume Dataset
- **Source:** Kaggle
- **Records:** 2,484
- **Target Variable:** Category
- **Number of Categories:** 24

Dataset columns:

- ID
- Resume_str
- Resume_html
- Category

---

# Technologies Used

## Machine Learning

- Python 3.11
- Scikit-learn
- TF-IDF Vectorizer
- Random Forest Classifier
- NumPy
- Pandas
- Joblib

## Backend

- FastAPI
- Uvicorn
- Pydantic

## Frontend

- Streamlit

## Database

- SQLite

## Resume Processing

- PyMuPDF
- python-docx

## Testing

- PyTest
- pytest-cov

## Code Quality

- Black
- isort
- Flake8

## Research

- Jupyter Notebook
- Matplotlib

---

# Machine Learning Pipeline

```
Resume Input
      │
      ▼
Security Validation
      │
      ▼
Resume Preprocessing
      │
      ▼
TF-IDF Feature Extraction
      │
      ▼
Random Forest Classifier
      │
      ▼
Prediction
      │
      ▼
SQLite Database
```

---

# Machine Learning Models Evaluated

The following machine learning models were evaluated during the research phase:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Random Forest Classifier
- Multinomial Naïve Bayes

### Selected Production Model

**Random Forest Classifier**

---

# Research vs Production

## Research Phase

- Jupyter Notebook
- Dataset exploration
- Text preprocessing
- Feature engineering
- Model comparison
- Performance evaluation

## Production Phase

- Modular package structure
- REST API
- Streamlit application
- SQLite persistence
- Logging
- Automated testing
- Security validation
- Model quality metrics
- Data quality metrics
- Production experimentation

---

# Software Engineering Features

The application implements the following software engineering practices:

- Modular Architecture
- Separation of Concerns
- REST API
- Input Validation
- Exception Handling
- Application Logging
- SQLite Persistence
- Automated Testing
- Code Formatting
- Static Code Analysis
- Model Quality Evaluation
- Data Quality Evaluation
- Production Deployment Experimentation

---

# System Architecture

The application consists of Machine Learning (ML) and Non-Machine Learning (Non-ML) components.

## Machine Learning Components

- Resume preprocessing
- TF-IDF feature extraction
- Random Forest classifier
- Prediction confidence
- Model quality metrics

## Non-Machine Learning Components

- Streamlit User Interface
- FastAPI REST API
- SQLite Database
- Security Validation
- Logging
- Testing Framework

---

# Application Architecture

```
                Streamlit UI
                     │
                     ▼
              FastAPI REST API
                     │
                     ▼
        Security & Input Validation
                     │
                     ▼
         Random Forest ML Model
                     │
                     ▼
          SQLite Prediction Database
```

---

# REST API

The application exposes REST endpoints using FastAPI.

Available endpoints:

- /
- /health
- /predict

Swagger Documentation:

```
http://127.0.0.1:8000/docs
```

---

# Logging

Centralized logging is implemented using Python's logging module.

Supported log levels:

- INFO
- WARNING
- ERROR

Logged events include:

- Application startup
- Prediction requests
- Validation warnings
- Model loading
- Database operations
- Unexpected exceptions

---

# Security Validation

The application validates all user input before prediction.

Implemented validation includes:

- Empty input validation
- Maximum length validation
- HTML/script detection
- SQL injection detection
- API key detection
- Credential detection
- Private key detection
- Suspicious input validation

---

# Automated Testing

The project includes multiple levels of automated testing.

A total of **13 automated tests** were successfully executed.

### Software Engineering Tests

- Unit Testing
- API Testing
- Integration Testing
- Data Validation Testing

### Machine Learning Component Tests

- Model training validation
- Inference validation
- Prediction invariance validation

---

# Model and Data Quality Metrics

## Model Quality Metrics

- Accuracy
- Precision
- Recall
- F1 Score

## Data Quality Metrics

- Missing Value Rate
- Duplicate Record Rate
- Schema Validation
- Class Distribution

---

# Production Deployment Experimentation

The project demonstrates production deployment strategies through simulation.

Implemented approaches:

- Shadow Deployment
- Canary Release
- A/B Testing

---

# Application Features

- Resume text prediction
- PDF resume upload
- DOCX resume upload
- TXT resume upload
- Prediction confidence
- Top prediction probabilities
- SQLite prediction history
- FastAPI REST API
- Swagger documentation
- Interactive Streamlit interface
- Application log viewer
- Model quality dashboard
- Data quality dashboard

---

# Project Structure

```text
resume-screening-se4ml-assignment2/

│
├── app/
│   ├── api/
│   ├── core/
│   ├── ml/
│   ├── security/
│   └── ui/
│
├── artifacts/
│   ├── production/
│   └── research/
│
├── data/
│
├── database/
│
├── logs/
│
├── models/
│   ├── production/
│   └── research/
│
├── notebooks/
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

---

# Reproducibility

## Prerequisites

- Python 3.11
- Conda
- Git

## Clone Repository

```bash
git clone https://github.com/Haridass-K/resume-screening-se4ml-assignment2.git

cd resume-screening-se4ml-assignment2
```

## Create Environment

```bash
conda create -n resume-screening python=3.11

conda activate resume-screening
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Code Quality Verification

```bash
black .

isort .

flake8 .

pytest -v
```

## Dataset

The Resume Dataset is not included due to repository size limitations.

Download the dataset from:

https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset

Place it in:

```
data/Resume.csv
```

## Model

The trained production model is included in:

models/production/resume_screening_model.pkl

The application can be executed directly without retraining.

---

# Running the Application

## Start FastAPI

```bash
uvicorn app.main:app --reload
```

Swagger:

```
http://127.0.0.1:8000/docs
```

---

## Launch Streamlit

```bash
streamlit run app/ui/streamlit_app.py
```

---

# Expected Output

The application enables users to:

- Predict resume category
- Upload PDF, DOCX, and TXT resumes
- View prediction confidence
- View top prediction probabilities
- Access REST APIs
- View Swagger documentation
- Monitor application logs
- View model quality metrics
- View data quality metrics
- Review production deployment experiments

---

# GitHub Repository

```
https://github.com/Haridass-K/resume-screening-se4ml-assignment2
```

---

# Group Details

**Group No:** 25

| BITS ID | Name | Email |
|----------|------|-------|
| 2024AC05325 | Haridass K | 2024ac05325@wilp.bits-pilani.ac.in |
| 2024AC05104 | Sathish T | 2024ac05104@wilp.bits-pilani.ac.in |
| 2024AC05651 | Tejaal M | 2024ac05651@wilp.bits-pilani.ac.in |
| 2024AC05728 | Sanjayan S | 2024ac05728@wilp.bits-pilani.ac.in |

---

# Conclusion

The **Production-Ready AI Resume Screening and Job Role Prediction System** demonstrates how software engineering principles can be successfully integrated into a machine learning application.

The project extends a research prototype into a modular production-ready system by implementing REST APIs, automated testing, logging, security validation, model and data quality evaluation, and production deployment experimentation while maintaining a scalable, maintainable, and reliable software architecture.

---

# Important Submission Notes

## Dataset

The dataset is excluded from this repository because of GitHub and LMS file size limitations.

Download it from:

https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset

Place it in:

```
data/Resume.csv
```

---

## Trained Model

The trained production model is included in:

```
models/production/
```

No retraining is required to execute the application.

---

## Complete Project

The GitHub repository includes:

- Production source code
- Research notebook
- FastAPI backend
- Streamlit frontend
- Production model
- Automated tests
- Documentation
- Requirements file

---
# Disclaimer

This project has been developed solely for academic purposes as part of the **Software Engineering for Machine Learning** course at **BITS Pilani Work Integrated Learning Programme (WILP)**. It is intended for educational demonstration and evaluation only and should not be used as a production recruitment or resume screening system.