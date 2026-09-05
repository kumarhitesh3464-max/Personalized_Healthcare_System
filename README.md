Personalized Healthcare & Medicine Recommendation System

A machine-learning-based educational healthcare application that analyzes selected symptoms, predicts a possible disease using a trained Random Forest model, and provides educational recommendations such as description, precautions, diet, medication information, and workout guidance.

Disclaimer: This project is for educational and demonstration purposes only. It is not a medical diagnosis system and must not be used as a substitute for a qualified healthcare professional.

Features

Symptom-based disease prediction

Random Forest classification model

342 symptom features

133 disease classes

Symptom normalization and feature alignment

Prediction confidence and confidence level

Top-5 disease predictions

Known/unknown symptom recognition

Integrated recommendation engine

Disease descriptions and precautions

Diet, medication, and workout information where available

Streamlit web interface

Safe dataset loading and validation

Technology Stack

Python

Streamlit

Pandas

NumPy

Scikit-learn

Joblib

Project Structure

Personalized_Healthcare_System/
│
├── app.py
├── README.md
├── requirements.txt
│
├── app/
│   ├── app.py
│   └── pages
│
├── assets/
├── data/
│   ├── dataset_1/
│   ├── dataset_2/
│   └── dataset_3/
│
├── models/
│   ├── disease_model.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
│
├── notebooks/
├── reports/
│
└── src/
    ├── disease_prediction.py
    ├── recommendation.py
    ├── preprocessing.py
    ├── data_loader.py
    └── utils.py

Datasets

Dataset 1

Contains the main disease-symptom training data and supporting information.

Important files:

dataset.csv

Symptom-severity.csv

symptom_Description.csv

symptom_precaution.csv

Dataset 2

Contains medical/patient data and is loaded by the data/recommendation layer.

Important file:

medical data.csv

Dataset 3

Provides additional recommendation information.

Important files:

description (2).csv

diets.csv

Diseases_and_Symptoms_dataset.csv

medications.csv

precautions.csv

workout.csv

Machine Learning Model

The application uses a trained RandomForestClassifier.

Current model information:

Model type: Random Forest Classifier

Features: 342

Disease classes: 133

Prediction: symptom feature vector → disease class

Probability: model predict_proba() output

Top results: highest-probability disease classes

The application reports the model confidence rather than automatically discarding low-probability predictions.

Prediction Workflow

User selects symptoms
        ↓
Symptom normalization
        ↓
Feature lookup
        ↓
342-feature input vector
        ↓
Random Forest model
        ↓
Disease probabilities
        ↓
Predicted disease + confidence
        ↓
Top-5 predictions
        ↓
Recommendation engine
        ↓
Educational healthcare information

Preprocessing

The preprocessing module performs:

Lowercase conversion

Underscore normalization

Hyphen/whitespace normalization

Disease-name normalization

Symptom extraction

Binary symptom feature creation

Feature alignment with the trained model

Duplicate pattern handling

Conflicting pattern detection

Example:

Fast__Heart_Rate
        ↓
fast heart rate

The prediction layer also normalizes feature names so variations such as underscores and hyphens can match the trained model.

Recommendation System

After prediction, the recommendation engine attempts to provide educational information from the available datasets.

Depending on data availability, results can include:

Disease description

Medication information

Diet recommendations

Precautions

Workout information

The recommendation engine is designed to fail safely when optional recommendation data is unavailable.

Running the Application

Open PowerShell in the project folder:

cd "C:\Users\HITESH\OneDrive\Desktop\Personalized_Healthcare_System"

Install dependencies:

pip install -r requirements.txt

Start Streamlit:

streamlit run app.py

Testing

The disease prediction module has been tested with:

abdominal pain

fast heart rate

fatigue, weight loss, polyuria

The tests verify:

Model loading

Label encoder loading

Feature count

Disease class count

Symptom recognition

Probability generation

Top-5 predictions

Recommendation engine availability

Important Notes

The trained model files in models/ are required for prediction.

Dataset files are required for the corresponding recommendation/data-loading features.

Do not rename model files unless source paths are updated.

Keep backup files separate from files imported by the application.

The main working Streamlit application currently runs from the root app.py; the empty app/app.py and app/pages entries are not required for the current working flow.

Limitations

Machine-learning predictions are probabilistic and may be incorrect.

A symptom can be associated with multiple diseases.

Low confidence does not prove that a disease is absent.

Dataset quality and coverage affect prediction quality.

Recommendations depend on available project data.

The system must not be used for emergency medical decisions.

Future Scope

Better symptom synonym handling

Improved model evaluation and validation

Additional medical datasets

Explainable AI

User history and prediction tracking

Improved UI/UX

More comprehensive recommendation coverage

Secure deployment

Medical-professional review of educational content

Academic Project

Personalized Healthcare & Medicine Recommendation System

Developed as an academic machine-learning and Streamlit p