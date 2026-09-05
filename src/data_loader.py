"""
Data Loader Module
Personalized Healthcare & Medicine Recommendation System

Responsible for loading all project datasets safely.
"""

import os
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

DATASET_1_DIR = os.path.join(DATA_DIR, "dataset_1")
DATASET_2_DIR = os.path.join(DATA_DIR, "dataset_2")
DATASET_3_DIR = os.path.join(DATA_DIR, "dataset_3")


# ---------------------------------------------------------
# GENERIC CSV LOADER
# ---------------------------------------------------------

def load_csv(file_path):
    """
    Load a CSV file and return a pandas DataFrame.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    try:
        df = pd.read_csv(file_path)

        # Remove completely empty rows
        df = df.dropna(how="all")

        return df

    except Exception as e:
        raise RuntimeError(
            f"Unable to load CSV file: {file_path}\nError: {e}"
        )


# ---------------------------------------------------------
# DATASET 1
# ---------------------------------------------------------

def load_dataset_1():
    """
    Load all files from Dataset 1.
    """

    data = {}

    data["main"] = load_csv(
        os.path.join(
            DATASET_1_DIR,
            "dataset.csv"
        )
    )

    data["severity"] = load_csv(
        os.path.join(
            DATASET_1_DIR,
            "Symptom-severity.csv"
        )
    )

    data["description"] = load_csv(
        os.path.join(
            DATASET_1_DIR,
            "symptom_Description.csv"
        )
    )

    data["precaution"] = load_csv(
        os.path.join(
            DATASET_1_DIR,
            "symptom_precaution.csv"
        )
    )

    return data


# ---------------------------------------------------------
# DATASET 2
# ---------------------------------------------------------

def load_dataset_2():
    """
    Load Dataset 2 medical patient data.
    """

    file_path = os.path.join(
        DATASET_2_DIR,
        "medical data.csv"
    )

    return load_csv(file_path)


# ---------------------------------------------------------
# DATASET 3
# ---------------------------------------------------------

def load_dataset_3():
    """
    Load all Dataset 3 files.
    """

    data = {}

    data["description"] = load_csv(
        os.path.join(
            DATASET_3_DIR,
            "description (2).csv"
        )
    )

    data["diets"] = load_csv(
        os.path.join(
            DATASET_3_DIR,
            "diets.csv"
        )
    )

    data["diseases_symptoms"] = load_csv(
        os.path.join(
            DATASET_3_DIR,
            "Diseases_and_Symptoms_dataset.csv"
        )
    )

    data["medications"] = load_csv(
        os.path.join(
            DATASET_3_DIR,
            "medications.csv"
        )
    )

    data["precautions"] = load_csv(
        os.path.join(
            DATASET_3_DIR,
            "precautions.csv"
        )
    )

    data["workout"] = load_csv(
        os.path.join(
            DATASET_3_DIR,
            "workout.csv"
        )
    )

    return data


# ---------------------------------------------------------
# LOAD ALL DATASETS
# ---------------------------------------------------------

def load_all_datasets():
    """
    Load Dataset 1, Dataset 2 and Dataset 3.

    Returns
    -------
    dict
        Dictionary containing all datasets.
    """

    return {
        "dataset_1": load_dataset_1(),
        "dataset_2": load_dataset_2(),
        "dataset_3": load_dataset_3()
    }


# ---------------------------------------------------------
# QUICK DATASET SUMMARY
# ---------------------------------------------------------

def get_dataset_summary(df):
    """
    Return basic information about a DataFrame.
    """

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }


# ---------------------------------------------------------
# MODULE TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("DATA LOADER TEST")
    print("=" * 60)

    try:

        datasets = load_all_datasets()

        print("\nDataset 1:")

        for name, df in datasets["dataset_1"].items():
            print(
                f"{name}: "
                f"{df.shape[0]} rows × "
                f"{df.shape[1]} columns"
            )

        print("\nDataset 2:")

        df2 = datasets["dataset_2"]

        print(
            f"medical data: "
            f"{df2.shape[0]} rows × "
            f"{df2.shape[1]} columns"
        )

        print("\nDataset 3:")

        for name, df in datasets["dataset_3"].items():
            print(
                f"{name}: "
                f"{df.shape[0]} rows × "
                f"{df.shape[1]} columns"
            )

        print("\n" + "=" * 60)
        print("DATA LOADING SUCCESSFUL")
        print("=" * 60)

    except Exception as e:

        print("\nDATA LOADING FAILED")
        print(e)