"""
Preprocessing Module
Personalized Healthcare & Medicine Recommendation System
"""

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def clean_text(value):
    """
    Normalize text values.

    Examples:
        ' Abdominal_Pain ' -> 'abdominal pain'
        'Fast__Heart_Rate' -> 'fast heart rate'
    """

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = value.replace("_", " ")

    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ---------------------------------------------------------
# CLEAN SINGLE SYMPTOM
# ---------------------------------------------------------

def normalize_symptom(symptom):
    """
    Normalize a symptom name.
    """

    return clean_text(symptom)


# ---------------------------------------------------------
# CLEAN DISEASE NAME
# ---------------------------------------------------------

def normalize_disease(disease):
    """
    Normalize disease names.
    """

    return clean_text(disease)


# ---------------------------------------------------------
# CLEAN DATAFRAME
# ---------------------------------------------------------

def clean_dataframe(df):
    """
    Clean object/string columns of a DataFrame.
    """

    df = df.copy()

    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = df[column].apply(clean_text)

    return df


# ---------------------------------------------------------
# CLEAN DATASET 1
# ---------------------------------------------------------

def clean_disease_symptom_dataset(df):
    """
    Clean Dataset 1 containing diseases and symptoms.
    """

    df = df.copy()

    if "Disease" in df.columns:

        df["Disease"] = df["Disease"].apply(
            normalize_disease
        )

    symptom_columns = [
        col
        for col in df.columns
        if str(col).lower().startswith("symptom")
    ]

    for column in symptom_columns:

        df[column] = df[column].apply(
            normalize_symptom
        )

    return df


# ---------------------------------------------------------
# GET UNIQUE SYMPTOMS
# ---------------------------------------------------------

def get_unique_symptoms(
    df,
    symptom_prefix="Symptom"
):
    """
    Extract all unique symptoms from symptom columns.
    """

    symptom_columns = [
        col
        for col in df.columns
        if str(col).lower().startswith(
            symptom_prefix.lower()
        )
    ]

    symptoms = set()

    for column in symptom_columns:

        for value in df[column]:

            value = normalize_symptom(value)

            if value:
                symptoms.add(value)

    return sorted(symptoms)


# ---------------------------------------------------------
# CREATE BINARY SYMPTOM FEATURES
# ---------------------------------------------------------

def create_symptom_features(df):
    """
    Convert symptom columns into binary symptom features.

    Example:

        fever = 1
        headache = 0
        cough = 1
    """

    df = clean_disease_symptom_dataset(df)

    if "Disease" not in df.columns:

        raise ValueError(
            "Dataset must contain a 'Disease' column."
        )

    symptoms = get_unique_symptoms(df)

    X = pd.DataFrame(
        0,
        index=df.index,
        columns=symptoms,
        dtype=np.int8
    )

    symptom_columns = [
        col
        for col in df.columns
        if str(col).lower().startswith("symptom")
    ]

    for column in symptom_columns:

        for index, symptom in df[column].items():

            symptom = normalize_symptom(symptom)

            if symptom in X.columns:

                X.loc[index, symptom] = 1

    y = df["Disease"].copy()

    return X, y


# ---------------------------------------------------------
# ALIGN FEATURES WITH TRAINED MODEL
# ---------------------------------------------------------

def align_features(
    X,
    feature_names
):
    """
    Align input features with the exact feature
    order expected by the trained model.
    """

    feature_names = [
        normalize_symptom(feature)
        for feature in feature_names
    ]

    X_aligned = pd.DataFrame(
        0,
        index=X.index,
        columns=feature_names,
        dtype=np.int8
    )

    common_features = [
        feature
        for feature in X.columns
        if feature in X_aligned.columns
    ]

    if common_features:

        X_aligned.loc[
            :,
            common_features
        ] = X.loc[
            :,
            common_features
        ]

    return X_aligned


# ---------------------------------------------------------
# CREATE INPUT FROM SELECTED SYMPTOMS
# ---------------------------------------------------------

def create_prediction_input(
    selected_symptoms,
    feature_names
):
    """
    Create a single-row ML input from selected symptoms.
    """

    normalized_features = [
        normalize_symptom(feature)
        for feature in feature_names
    ]

    normalized_selected = {
        normalize_symptom(symptom)
        for symptom in selected_symptoms
    }

    X = pd.DataFrame(
        0,
        index=[0],
        columns=normalized_features,
        dtype=np.int8
    )

    recognized = []

    unknown = []

    for symptom in normalized_selected:

        if symptom in X.columns:

            X.loc[0, symptom] = 1

            recognized.append(symptom)

        else:

            unknown.append(symptom)

    return X, sorted(recognized), sorted(unknown)


# ---------------------------------------------------------
# REMOVE DUPLICATE PATTERNS
# ---------------------------------------------------------

def remove_duplicate_patterns(X, y):
    """
    Remove duplicate symptom patterns while keeping
    the associated disease label.

    Returns:
        X_unique
        y_unique
    """

    combined = pd.concat(
        [
            X.reset_index(drop=True),
            y.reset_index(drop=True).rename(
                "Disease"
            )
        ],
        axis=1
    )

    combined = combined.drop_duplicates()

    X_unique = combined.drop(
        columns=["Disease"]
    )

    y_unique = combined["Disease"]

    return X_unique, y_unique


# ---------------------------------------------------------
# CHECK CONFLICTING PATTERNS
# ---------------------------------------------------------

def find_conflicting_patterns(X, y):
    """
    Find identical symptom patterns associated with
    multiple diseases.
    """

    combined = pd.concat(
        [
            X.reset_index(drop=True),
            y.reset_index(drop=True).rename(
                "Disease"
            )
        ],
        axis=1
    )

    feature_columns = [
        col
        for col in combined.columns
        if col != "Disease"
    ]

    conflicts = (
        combined
        .groupby(feature_columns)["Disease"]
        .nunique()
    )

    return conflicts[conflicts > 1]


# ---------------------------------------------------------
# DATASET PREPROCESSING PIPELINE
# ---------------------------------------------------------

def preprocess_dataset(df):
    """
    Complete preprocessing pipeline for disease/symptom data.
    """

    cleaned_df = clean_disease_symptom_dataset(df)

    X, y = create_symptom_features(
        cleaned_df
    )

    X_unique, y_unique = remove_duplicate_patterns(
        X,
        y
    )

    return {
        "cleaned_data": cleaned_df,
        "X": X,
        "y": y,
        "X_unique": X_unique,
        "y_unique": y_unique,
        "feature_names": X.columns.tolist()
    }


# ---------------------------------------------------------
# MODULE TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("PREPROCESSING MODULE TEST")
    print("=" * 60)

    print("\nTesting text normalization:")

    examples = [
        " Abdominal_Pain ",
        "Fast__Heart_Rate",
        "  Weight_Gain  ",
        "Acute Liver Failure"
    ]

    for value in examples:

        print(
            f"{value!r} -> "
            f"{clean_text(value)!r}"
        )

    print("\nPreprocessing module loaded successfully.")

    print("=" * 60)