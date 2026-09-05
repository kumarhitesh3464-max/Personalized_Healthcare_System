"""
Utility Functions
Personalized Healthcare & Medicine Recommendation System
"""

import os
import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ---------------------------------------------------------
# FILE UTILITIES
# ---------------------------------------------------------

def file_exists(file_path):
    """
    Check whether a file exists.
    """

    return os.path.isfile(file_path)


def directory_exists(directory_path):
    """
    Check whether a directory exists.
    """

    return os.path.isdir(directory_path)


def ensure_directory(directory_path):
    """
    Create directory if it does not exist.
    """

    os.makedirs(
        directory_path,
        exist_ok=True
    )

    return directory_path


# ---------------------------------------------------------
# CONFIDENCE UTILITIES
# ---------------------------------------------------------

def confidence_percentage(confidence):
    """
    Convert probability to percentage.
    """

    try:

        confidence = float(confidence)

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        return round(
            confidence * 100,
            2
        )

    except (TypeError, ValueError):

        return 0.0


def get_confidence_level(confidence):
    """
    Convert probability into a readable confidence level.
    """

    try:

        confidence = float(confidence)

    except (TypeError, ValueError):

        return "Unknown"

    if confidence >= 0.80:

        return "High"

    elif confidence >= 0.50:

        return "Moderate"

    elif confidence >= 0.30:

        return "Low"

    else:

        return "Very Low"


# ---------------------------------------------------------
# SAFE VALUE CONVERSION
# ---------------------------------------------------------

def safe_float(value, default=0.0):
    """
    Safely convert value to float.
    """

    try:

        return float(value)

    except (
        TypeError,
        ValueError
    ):

        return default


def safe_int(value, default=0):
    """
    Safely convert value to integer.
    """

    try:

        return int(value)

    except (
        TypeError,
        ValueError
    ):

        return default


# ---------------------------------------------------------
# TEXT UTILITIES
# ---------------------------------------------------------

def clean_string(value):
    """
    Convert a value into a clean string.
    """

    if value is None:

        return ""

    if pd.isna(value):

        return ""

    return str(value).strip()


def normalize_whitespace(value):
    """
    Remove unnecessary whitespace.
    """

    value = clean_string(value)

    return " ".join(
        value.split()
    )


# ---------------------------------------------------------
# LIST UTILITIES
# ---------------------------------------------------------

def unique_list(values):
    """
    Return unique non-empty values while preserving order.
    """

    if values is None:

        return []

    result = []

    seen = set()

    for value in values:

        value = clean_string(value)

        if not value:
            continue

        key = value.lower()

        if key not in seen:

            seen.add(key)

            result.append(value)

    return result


# ---------------------------------------------------------
# JSON SAFE CONVERSION
# ---------------------------------------------------------

def make_json_serializable(value):
    """
    Convert NumPy/Pandas objects into JSON-compatible values.
    """

    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32
        )
    ):

        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32
        )
    ):

        return float(value)

    if isinstance(
        value,
        np.ndarray
    ):

        return [
            make_json_serializable(item)
            for item in value.tolist()
        ]

    if isinstance(
        value,
        pd.Series
    ):

        return {
            str(key):
            make_json_serializable(val)
            for key, val in value.items()
        }

    if isinstance(
        value,
        pd.DataFrame
    ):

        return [
            make_json_serializable(row)
            for row in value.to_dict(
                orient="records"
            )
        ]

    if isinstance(
        value,
        dict
    ):

        return {
            str(key):
            make_json_serializable(val)
            for key, val in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set)
    ):

        return [
            make_json_serializable(item)
            for item in value
        ]

    return value


# ---------------------------------------------------------
# SAVE JSON
# ---------------------------------------------------------

def save_json(
    data,
    file_path
):
    """
    Save Python object as JSON.
    """

    directory = os.path.dirname(
        os.path.abspath(file_path)
    )

    ensure_directory(directory)

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            make_json_serializable(data),
            file,
            indent=4,
            ensure_ascii=False
        )


# ---------------------------------------------------------
# LOAD JSON
# ---------------------------------------------------------

def load_json(file_path):
    """
    Load JSON file.
    """

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"JSON file not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ---------------------------------------------------------
# MODEL RESULT FORMATTER
# ---------------------------------------------------------

def format_prediction_result(
    disease,
    confidence,
    recognized_symptoms=None,
    unknown_symptoms=None
):
    """
    Create a standard prediction result dictionary.
    """

    if recognized_symptoms is None:

        recognized_symptoms = []

    if unknown_symptoms is None:

        unknown_symptoms = []

    confidence = safe_float(
        confidence
    )

    return {

        "disease": clean_string(
            disease
        ),

        "confidence": confidence,

        "confidence_percentage":
            confidence_percentage(
                confidence
            ),

        "confidence_level":
            get_confidence_level(
                confidence
            ),

        "recognized_symptoms":
            len(recognized_symptoms),

        "recognized_symptoms_list":
            unique_list(
                recognized_symptoms
            ),

        "unknown_symptoms":
            unique_list(
                unknown_symptoms
            )
    }


# ---------------------------------------------------------
# PROJECT DIRECTORY CHECK
# ---------------------------------------------------------

def get_project_directories():
    """
    Return important project directories.
    """

    directories = {

        "base": BASE_DIR,

        "data": os.path.join(
            BASE_DIR,
            "data"
        ),

        "models": os.path.join(
            BASE_DIR,
            "models"
        ),

        "src": os.path.join(
            BASE_DIR,
            "src"
        ),

        "notebooks": os.path.join(
            BASE_DIR,
            "notebooks"
        ),

        "reports": os.path.join(
            BASE_DIR,
            "reports"
        ),

        "assets": os.path.join(
            BASE_DIR,
            "assets"
        )
    }

    return directories


# ---------------------------------------------------------
# MODULE TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("UTILS MODULE TEST")
    print("=" * 60)

    print(
        "\nConfidence:",
        confidence_percentage(0.8734),
        "%"
    )

    print(
        "Confidence Level:",
        get_confidence_level(0.8734)
    )

    print(
        "\nUnique List:",
        unique_list(
            [
                "Fever",
                "fever",
                "Cough",
                "",
                "Cough"
            ]
        )
    )

    result = format_prediction_result(
        disease="Diabetes",
        confidence=0.91,
        recognized_symptoms=[
            "polyuria",
            "polydipsia"
        ],
        unknown_symptoms=[]
    )

    print("\nSample Prediction Result:")

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    print("\nProject Directories:")

    for name, path in get_project_directories().items():

        print(
            f"{name}: {path}"
        )

    print("\nUTILS MODULE TEST SUCCESSFUL")

    print("=" * 60)