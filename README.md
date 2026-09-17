# 🏥 Personalized Healthcare & Medicine Recommendation System

> An end-to-end Machine Learning and Streamlit-based healthcare application that analyzes user symptoms, predicts a possible disease, and provides educational healthcare recommendations.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Render-success?style=for-the-badge)](https://personalized-healthcare-system.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square\&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square\&logo=streamlit)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?style=flat-square\&logo=scikit-learn)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-purple?style=flat-square)](https://render.com/)

---

## 🌐 Live Application

### 🚀 Try the Project Online

**Live Demo:**
https://personalized-healthcare-system.onrender.com

The application is deployed as a Streamlit web application on Render.

Users can select symptoms through an interactive interface and receive a machine-learning-based prediction along with available educational healthcare information.

---

# 📌 Project Overview

The **Personalized Healthcare & Medicine Recommendation System** is an educational Machine Learning project designed to demonstrate how symptom-based disease prediction can be integrated with a user-friendly healthcare recommendation interface.

The system uses a trained **Random Forest Classifier** to analyze symptom combinations and predict one of the diseases represented in the training data.

Based on the prediction and available project data, the application can display:

* 🩺 Predicted disease
* 📊 Model confidence
* 📈 Top disease probability analysis
* 🔎 Recognized and unrecognized symptoms
* 🧠 Model feature importance
* 📖 Disease description
* 💊 Medication reference information
* 🥗 Diet recommendations
* ⚠️ Precautions
* 🏃 Workout/lifestyle information

The application also distinguishes **health categories such as Weight Loss and Weight Gain from disease predictions**, preventing them from being presented as diseases.

---

# 🎯 Objectives

The major objectives of this project are:

1. Build a machine-learning-based symptom analysis system.
2. Predict diseases from user-selected symptoms.
3. Create an interactive healthcare interface using Streamlit.
4. Provide educational disease-related recommendations.
5. Visualize model confidence and probability information.
6. Show model feature importance for basic explainability.
7. Deploy the complete application on a cloud platform.
8. Demonstrate an end-to-end ML workflow from data preprocessing to deployment.

---

# ✨ Key Features

| Feature                     | Description                                                            |
| --------------------------- | ---------------------------------------------------------------------- |
| 🩺 Symptom Selection        | Interactive symptom selection interface                                |
| 🤖 Disease Prediction       | Random Forest-based disease prediction                                 |
| 📊 Confidence Score         | Displays model prediction confidence                                   |
| 📈 Probability Analysis     | Shows top predicted diseases and probabilities                         |
| 🔎 Symptom Recognition      | Identifies recognized and unrecognized symptoms                        |
| 🧠 Explainability           | Displays important symptoms/features used by the model                 |
| 📖 Disease Information      | Provides available disease descriptions                                |
| 💊 Medication Reference     | Displays educational medication information when available             |
| 🥗 Diet Guidance            | Shows available disease-specific dietary information                   |
| ⚠️ Precautions              | Displays available precaution information                              |
| 🏃 Lifestyle/Workout        | Provides available lifestyle or workout guidance                       |
| ⚖️ Weight Category Handling | Weight Gain/Loss are treated as health categories rather than diseases |
| 🌐 Cloud Deployment         | Deployed using Render                                                  |
| 📱 Responsive UI            | Streamlit-based interactive interface                                  |

---

# 🧠 Machine Learning

## Model

The project uses a:

**Random Forest Classifier**

The trained model is serialized using `joblib` and loaded by the Streamlit application during runtime.

### Model Configuration

```text
Algorithm       : Random Forest Classifier
Features        : 342
Disease Classes : 133
Model Size      : ~0.87 MB
```

### Model Evaluation

| Metric   |  Score |
| -------- | -----: |
| Accuracy | 51.19% |
| F1 Score | 54.65% |

> These metrics represent the performance of the trained model on the project's evaluation data. They should not be interpreted as clinical diagnostic accuracy.

---

# 🔄 System Workflow

```text
                 ┌─────────────────────┐
                 │      User Input      │
                 │   Select Symptoms    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Symptom Processing  │
                 │ & Normalization     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Feature Preparation │
                 │   342 Features      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Random Forest ML   │
                 │      Model          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Disease Prediction  │
                 │ + Confidence Score  │
                 └──────────┬──────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
       ┌────────────────┐      ┌──────────────────┐
       │ Probability &  │      │ Healthcare       │
       │ Explainability │      │ Recommendations  │
       └────────────────┘      └──────────────────┘
```

---

# 🏗️ Project Architecture

```text
Personalized_Healthcare_System/
│
├── app.py
│
├── src/
│   ├── data_loader.py
│   ├── disease_prediction.py
│   ├── preprocessing.py
│   ├── recommendation.py
│   └── utils.py
│
├── models/
│   ├── disease_model.pkl
│   ├── feature_names.pkl
│   └── label_encoder.pkl
│
├── data/
│   └── Medical datasets
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_traning.ipynb
│   └── 04_model_evaluation.ipynb
│
├── reports/
│   └── Project reports and documentation
│
├── assets/
│   └── Application assets
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Technology Stack

### Programming

* Python

### Machine Learning

* Scikit-learn
* Random Forest
* Joblib

### Data Processing

* Pandas
* NumPy

### Application

* Streamlit

### Development & Analysis

* Jupyter Notebook

### Deployment

* Render
* GitHub

---

# 📚 Project Workflow

## 1. Data Analysis

The dataset is explored to understand:

* Dataset structure
* Missing values
* Symptom distribution
* Disease distribution
* Feature relationships

---

## 2. Data Preprocessing

The preprocessing stage includes:

* Cleaning symptom data
* Standardizing feature names
* Preparing model-compatible features
* Encoding disease labels
* Creating the final feature matrix

---

## 3. Model Training

A Random Forest classifier is trained using the processed symptom dataset.

The trained model is then saved using Joblib.

---

## 4. Model Evaluation

The trained model is evaluated using metrics including:

* Accuracy
* F1 Score
* Prediction probabilities

---

## 5. Streamlit Application

The trained model is integrated into a Streamlit interface where users can:

1. Select symptoms.
2. Submit the symptoms.
3. Receive the model prediction.
4. View confidence information.
5. Analyze top predictions.
6. Explore symptom recognition.
7. View model feature importance.
8. Access available educational healthcare recommendations.

---

# 🚀 Run Locally

## Prerequisites

Make sure Python 3.x is installed.

Check Python:

```bash
python --version
```

---

## Clone Repository

```bash
git clone https://github.com/kumarhitesh3464-max/Personalized_Healthcare_System.git
```

Move into the project:

```bash
cd Personalized_Healthcare_System
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

The project's dependencies are maintained through `requirements.txt`, which is also the standard mechanism used to install Python packages during deployment.

---

## Run Application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# ☁️ Deployment

The application is deployed on **Render**.

### Deployment Configuration

```text
Build Command:
pip install -r requirements.txt

Start Command:
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

### Live URL

```text
https://personalized-healthcare-system.onrender.com
```

---

# 📊 Example Application Flow

```text
User selects symptoms
        ↓
Symptoms are normalized
        ↓
Model-compatible features are created
        ↓
Random Forest generates prediction
        ↓
Confidence is calculated
        ↓
Top predictions are analyzed
        ↓
Healthcare information is displayed
```

---

# 🔍 Explainability

The application provides basic model explainability using **feature importance** from the Random Forest model.

The interface displays the symptoms/features with the highest overall importance across the model's decision trees.

This helps users understand which features have greater influence within the trained model.

> Feature importance shows model behavior and does not establish medical causation.

---

# ⚖️ Weight Gain & Weight Loss Handling

Weight changes are treated separately from disease predictions.

For example:

```text
Weight Loss
     ↓
Health Category: Weight Loss
```

instead of incorrectly presenting:

```text
Predicted Disease: Jaundice
```

Similarly:

```text
Weight Gain
     ↓
Health Category: Weight Gain
```

This distinction helps prevent a symptom/category from being displayed as a confirmed disease.

---

# 🔐 Safety & Medical Disclaimer

⚠️ **IMPORTANT**

This application is an **educational Machine Learning project**.

It is **not a medical diagnostic system**, and its predictions should not be considered a professional medical diagnosis.

The application may provide educational references related to:

* Diseases
* Medication
* Diet
* Precautions
* Workout/lifestyle

These recommendations should **not** be used for self-medication or treatment decisions.

Always consult a qualified healthcare professional for personalized medical advice.

---

# ⚠️ Model Limitations

This project has several important limitations:

* The model is trained on the available project datasets.
* Model performance depends on the quality and distribution of the training data.
* A symptom can be associated with multiple diseases.
* Predictions may be incorrect.
* Confidence scores represent model probabilities, not clinical certainty.
* The system does not replace professional medical examination.
* The model should be considered a demonstration/prototype rather than a clinically validated system.

---

# 🎓 Educational Purpose

This project demonstrates practical implementation of:

* Machine Learning
* Classification
* Data preprocessing
* Feature engineering
* Model evaluation
* Probability estimation
* Explainable ML concepts
* Streamlit application development
* Cloud deployment
* Git/GitHub workflow

It can be used as an academic project and as a demonstration of an end-to-end Machine Learning application.

---

# 📁 Important Files

| File                                    | Purpose                         |
| --------------------------------------- | ------------------------------- |
| `app.py`                                | Main Streamlit application      |
| `src/disease_prediction.py`             | Disease prediction logic        |
| `src/preprocessing.py`                  | Data preprocessing              |
| `src/recommendation.py`                 | Healthcare recommendation logic |
| `src/data_loader.py`                    | Dataset/model data loading      |
| `models/disease_model.pkl`              | Trained Random Forest model     |
| `models/feature_names.pkl`              | Model feature names             |
| `models/label_encoder.pkl`              | Disease label encoder           |
| `notebooks/01_data_analysis.ipynb`      | Exploratory data analysis       |
| `notebooks/02_data_preprocessing.ipynb` | Data preprocessing              |
| `notebooks/03_model_traning.ipynb`      | Model training                  |
| `notebooks/04_model_evaluation.ipynb`   | Model evaluation                |
| `requirements.txt`                      | Python dependencies             |

---

# 🔮 Future Improvements

Potential future improvements include:

* Larger and more diverse medical datasets
* Improved model accuracy
* Hyperparameter optimization
* Cross-validation
* Better handling of class imbalance
* More advanced explainability using SHAP
* Medical knowledge-base integration
* Natural-language symptom input
* Multilingual support
* User authentication
* Patient history tracking
* Doctor consultation integration
* Improved recommendation validation
* Automated model monitoring
* Containerized deployment using Docker

---

# 🤝 Contributing

Contributions and suggestions are welcome.

To contribute:

```bash
git clone https://github.com/kumarhitesh3464-max/Personalized_Healthcare_System.git
cd Personalized_Healthcare_System
```

Create a new branch:

```bash
git checkout -b feature/new-feature
```

Make your changes, commit them, and push the branch.

---

# 📌 Project Status

**Status:** 🟢 Deployed

**Application:** Streamlit

**Deployment:** Render

**Model:** Random Forest Classifier

**Disease Classes:** 133

**Features:** 342

---

# 👨‍💻 Author

**Hitesh Kumar**

### Project

**Personalized Healthcare & Medicine Recommendation System**

### Live Application

🚀 https://personalized-healthcare-system.onrender.com

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub and sharing feedback.

---

> **Disclaimer:** This project is developed for educational and academic purposes only. It is not intended to provide medical diagnosis, treatment, or professional healthcare advice.
