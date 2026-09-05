import os
import ast
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

DATASET_1_DIR = os.path.join(
    DATA_DIR,
    "dataset_1"
)

DATASET_2_DIR = os.path.join(
    DATA_DIR,
    "dataset_2"
)

DATASET_3_DIR = os.path.join(
    DATA_DIR,
    "dataset_3"
)


# ============================================================
# FILE PATHS
# ============================================================

DISEASE_DESCRIPTION_1 = os.path.join(
    DATASET_1_DIR,
    "symptom_Description.csv"
)

DISEASE_PRECAUTION_1 = os.path.join(
    DATASET_1_DIR,
    "symptom_precaution.csv"
)

MEDICAL_DATA_2 = os.path.join(
    DATASET_2_DIR,
    "medical data.csv"
)

DISEASE_DESCRIPTION_3 = os.path.join(
    DATASET_3_DIR,
    "description (2).csv"
)

DISEASE_DIET_3 = os.path.join(
    DATASET_3_DIR,
    "diets.csv"
)

DISEASE_MEDICATION_3 = os.path.join(
    DATASET_3_DIR,
    "medications.csv"
)

DISEASE_PRECAUTION_3 = os.path.join(
    DATASET_3_DIR,
    "precautions.csv"
)

DISEASE_WORKOUT_3 = os.path.join(
    DATASET_3_DIR,
    "workout.csv"
)

DISEASE_MAPPING_FILE = os.path.join(
    DATA_DIR,
    "disease_mapping.csv"
)

DISEASE_RECOMMENDATION_FILE = os.path.join(
    DATA_DIR,
    "disease_recommendations.csv"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize disease/recommendation text.

    Examples:

        "Heart-Attack"
        "heart_attack"
        " Heart Attack "

    become:

        "heart attack"
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    value = str(value).strip().lower()

    value = value.replace("_", " ")
    value = value.replace("-", " ")

    value = value.replace("(", "")
    value = value.replace(")", "")

    value = " ".join(
        value.split()
    )

    return value


# ============================================================
# CLEAN VALUE
# ============================================================

def clean_value(value):
    """
    Convert a dataframe value into clean text.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    if not text:
        return ""

    if text.lower() in {
        "nan",
        "none",
        "null",
        "na",
        "n/a"
    }:
        return ""

    return text


# ============================================================
# SAFE LIST PARSER
# ============================================================

def parse_recommendation_value(value):
    """
    Convert a CSV cell into a list of clean values.

    Supported formats:

        normal text
        [item1, item2]
        ['item1', 'item2']
        comma separated text
        semicolon separated text
    """

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    text = str(value).strip()

    if not text:
        return []

    if text.lower() in {
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "[]"
    }:
        return []

    # --------------------------------------------------------
    # Python list format
    # --------------------------------------------------------

    if (
        text.startswith("[")
        and text.endswith("]")
    ):

        try:

            parsed = ast.literal_eval(
                text
            )

            if isinstance(
                parsed,
                (list, tuple, set)
            ):

                result = []

                for item in parsed:

                    item = clean_value(
                        item
                    )

                    if item:
                        result.append(item)

                return unique_values(
                    result
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # Semicolon separated
    # --------------------------------------------------------

    if ";" in text:

        values = text.split(";")

    # --------------------------------------------------------
    # Comma separated
    # --------------------------------------------------------

    elif "," in text:

        values = text.split(",")

    # --------------------------------------------------------
    # Single value
    # --------------------------------------------------------

    else:

        values = [text]

    result = []

    for item in values:

        item = clean_value(
            item
        )

        if item:
            result.append(item)

    return unique_values(
        result
    )


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def unique_values(values):
    """
    Remove duplicate values while preserving order.
    """

    result = []

    seen = set()

    for value in values:

        value = clean_value(
            value
        )

        if not value:
            continue

        key = normalize_text(
            value
        )

        if not key:
            continue

        if key not in seen:

            seen.add(key)

            result.append(
                value
            )

    return result


# ============================================================
# SAFE CSV LOADER
# ============================================================

def load_csv(path):
    """
    Load CSV safely.

    Missing optional recommendation files do not crash
    the complete application.
    """

    if not os.path.exists(path):

        print(
            f"Warning: CSV file not found: {path}"
        )

        return pd.DataFrame()

    try:

        dataframe = pd.read_csv(
            path
        )

        return dataframe

    except Exception as error:

        print(
            f"Warning: Could not load CSV: {path}"
        )

        print(
            f"Reason: {error}"
        )

        return pd.DataFrame()


# ============================================================
# LOAD ALL DATASETS
# ============================================================

df_description_1 = load_csv(
    DISEASE_DESCRIPTION_1
)

df_precaution_1 = load_csv(
    DISEASE_PRECAUTION_1
)

df_medical_2 = load_csv(
    MEDICAL_DATA_2
)

df_description_3 = load_csv(
    DISEASE_DESCRIPTION_3
)

df_diet_3 = load_csv(
    DISEASE_DIET_3
)

df_medication_3 = load_csv(
    DISEASE_MEDICATION_3
)

df_precaution_3 = load_csv(
    DISEASE_PRECAUTION_3
)

df_workout_3 = load_csv(
    DISEASE_WORKOUT_3
)

df_mapping = load_csv(
    DISEASE_MAPPING_FILE
)

df_recommendations = load_csv(
    DISEASE_RECOMMENDATION_FILE
)


# ============================================================
# NORMALIZE DISEASE COLUMN
# ============================================================

def normalize_dataframe_disease(
    dataframe,
    possible_columns
):
    """
    Create _normalized_disease column from
    the first available disease column.
    """

    if dataframe.empty:
        return dataframe

    for column in possible_columns:

        if column in dataframe.columns:

            dataframe = dataframe.copy()

            dataframe[
                "_normalized_disease"
            ] = dataframe[
                column
            ].apply(
                normalize_text
            )

            return dataframe

    return dataframe


# ============================================================
# APPLY DISEASE NORMALIZATION
# ============================================================

df_description_1 = normalize_dataframe_disease(
    df_description_1,
    [
        "Disease",
        "disease"
    ]
)

df_precaution_1 = normalize_dataframe_disease(
    df_precaution_1,
    [
        "Disease",
        "disease"
    ]
)

df_medical_2 = normalize_dataframe_disease(
    df_medical_2,
    [
        "Disease",
        "disease"
    ]
)

df_description_3 = normalize_dataframe_disease(
    df_description_3,
    [
        "Disease",
        "disease"
    ]
)

df_diet_3 = normalize_dataframe_disease(
    df_diet_3,
    [
        "Disease",
        "disease"
    ]
)

df_medication_3 = normalize_dataframe_disease(
    df_medication_3,
    [
        "Disease",
        "disease"
    ]
)

df_precaution_3 = normalize_dataframe_disease(
    df_precaution_3,
    [
        "Disease",
        "disease"
    ]
)

df_workout_3 = normalize_dataframe_disease(
    df_workout_3,
    [
        "Disease",
        "disease"
    ]
)


# ============================================================
# VERIFIED DISEASE MAPPING
# ============================================================

mapping_lookup = {}


if not df_mapping.empty:

    required_columns = {
        "model_disease",
        "dataset3_disease"
    }

    if required_columns.issubset(
        df_mapping.columns
    ):

        for _, row in (
            df_mapping.iterrows()
        ):

            model_disease = normalize_text(
                row[
                    "model_disease"
                ]
            )

            dataset3_disease = normalize_text(
                row[
                    "dataset3_disease"
                ]
            )

            if (
                model_disease
                and
                dataset3_disease
            ):

                mapping_lookup[
                    model_disease
                ] = dataset3_disease


# ============================================================
# DISEASE RECOMMENDATION LOOKUP
# ============================================================

recommendation_lookup = {}


if not df_recommendations.empty:

    for _, row in (
        df_recommendations.iterrows()
    ):

        if "disease" not in row.index:
            continue

        disease_key = normalize_text(
            row[
                "disease"
            ]
        )

        if not disease_key:
            continue

        if disease_key not in (
            recommendation_lookup
        ):

            recommendation_lookup[
                disease_key
            ] = {
                "diet": [],
                "workout": []
            }

        # ----------------------------------------------------
        # DIET
        # ----------------------------------------------------

        if "diet" in row.index:

            recommendation_lookup[
                disease_key
            ][
                "diet"
            ].extend(
                parse_recommendation_value(
                    row[
                        "diet"
                    ]
                )
            )

        # ----------------------------------------------------
        # WORKOUT
        # ----------------------------------------------------

        if "workout" in row.index:

            recommendation_lookup[
                disease_key
            ][
                "workout"
            ].extend(
                parse_recommendation_value(
                    row[
                        "workout"
                    ]
                )
            )


# ============================================================
# CLEAN RECOMMENDATION LOOKUP
# ============================================================

for disease_key in recommendation_lookup:

    recommendation_lookup[
        disease_key
    ][
        "diet"
    ] = unique_values(
        recommendation_lookup[
            disease_key
        ][
            "diet"
        ]
    )

    recommendation_lookup[
        disease_key
    ][
        "workout"
    ] = unique_values(
        recommendation_lookup[
            disease_key
        ][
            "workout"
        ]
    )


# ============================================================
# DATASET-3 MAPPING
# ============================================================

def get_dataset3_disease(
    model_disease
):
    """
    Return Dataset-3 disease only when an
    explicit verified mapping exists.
    """

    disease_key = normalize_text(
        model_disease
    )

    if not disease_key:
        return None

    return mapping_lookup.get(
        disease_key
    )


# ============================================================
# GENERIC DISEASE RECORD LOOKUP
# ============================================================

def find_disease_record(
    dataframe,
    disease
):
    """
    Return the first exact normalized disease match.
    """

    if dataframe.empty:
        return None

    if (
        "_normalized_disease"
        not in dataframe.columns
    ):
        return None

    disease_key = normalize_text(
        disease
    )

    if not disease_key:
        return None

    matches = dataframe[
        dataframe[
            "_normalized_disease"
        ]
        == disease_key
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


# ============================================================
# ALL DISEASE RECORDS
# ============================================================

def find_all_disease_records(
    dataframe,
    disease
):
    """
    Return all rows belonging to the disease.
    """

    if dataframe.empty:
        return pd.DataFrame()

    if (
        "_normalized_disease"
        not in dataframe.columns
    ):
        return pd.DataFrame()

    disease_key = normalize_text(
        disease
    )

    if not disease_key:
        return pd.DataFrame()

    return dataframe[
        dataframe[
            "_normalized_disease"
        ]
        == disease_key
    ]


# ============================================================
# DESCRIPTION
# ============================================================

def get_description(
    disease
):

    # --------------------------------------------------------
    # Dataset 1
    # --------------------------------------------------------

    row = find_disease_record(
        df_description_1,
        disease
    )

    if row is not None:

        for column in [
            "Description",
            "description"
        ]:

            if column in row.index:

                text = clean_value(
                    row[
                        column
                    ]
                )

                if text:
                    return text


    # --------------------------------------------------------
    # Dataset 3 through verified mapping
    # --------------------------------------------------------

    mapped_disease = get_dataset3_disease(
        disease
    )

    if mapped_disease:

        row = find_disease_record(
            df_description_3,
            mapped_disease
        )

        if row is not None:

            for column in [
                "Description",
                "description"
            ]:

                if column in row.index:

                    text = clean_value(
                        row[
                            column
                        ]
                    )

                    if text:
                        return text


    return (
        "No detailed disease description "
        "is available in the connected datasets."
    )


# ============================================================
# PRECAUTIONS
# ============================================================

def get_precautions(
    disease
):

    precautions = []


    # --------------------------------------------------------
    # Dataset 1
    # --------------------------------------------------------

    row = find_disease_record(
        df_precaution_1,
        disease
    )

    if row is not None:

        for column in row.index:

            if (
                "precaution"
                in str(column).lower()
            ):

                value = clean_value(
                    row[
                        column
                    ]
                )

                if value:

                    precautions.extend(
                        parse_recommendation_value(
                            value
                        )
                    )


    precautions = unique_values(
        precautions
    )

    if precautions:
        return precautions


    # --------------------------------------------------------
    # Dataset 3 through verified mapping
    # --------------------------------------------------------

    mapped_disease = get_dataset3_disease(
        disease
    )

    if mapped_disease:

        matched_rows = (
            find_all_disease_records(
                df_precaution_3,
                mapped_disease
            )
        )

        if not matched_rows.empty:

            for _, row in (
                matched_rows.iterrows()
            ):

                for column in row.index:

                    if (
                        column
                        == "_normalized_disease"
                    ):
                        continue

                    if (
                        "precaution"
                        not in str(column).lower()
                    ):
                        continue

                    value = clean_value(
                        row[
                            column
                        ]
                    )

                    if value:

                        precautions.extend(
                            parse_recommendation_value(
                                value
                            )
                        )


    precautions = unique_values(
        precautions
    )

    if precautions:
        return precautions


    return [
        "Consult a qualified healthcare professional "
        "for personalized medical advice."
    ]


# ============================================================
# MEDICATIONS
# ============================================================

def get_medications(
    disease
):

    medications = []


    # --------------------------------------------------------
    # Dataset 2
    # --------------------------------------------------------

    row = find_disease_record(
        df_medical_2,
        disease
    )

    if row is not None:

        for column in [
            "Medicine",
            "medicine",
            "Medication",
            "medication",
            "Medicines",
            "medicines"
        ]:

            if column in row.index:

                value = clean_value(
                    row[
                        column
                    ]
                )

                if value:

                    medications.extend(
                        parse_recommendation_value(
                            value
                        )
                    )


    medications = unique_values(
        medications
    )

    if medications:
        return medications


    # --------------------------------------------------------
    # Dataset 3 through verified mapping
    # --------------------------------------------------------

    mapped_disease = get_dataset3_disease(
        disease
    )

    if mapped_disease:

        matched_rows = (
            find_all_disease_records(
                df_medication_3,
                mapped_disease
            )
        )

        if not matched_rows.empty:

            for _, row in (
                matched_rows.iterrows()
            ):

                for column in row.index:

                    if (
                        column
                        == "_normalized_disease"
                    ):
                        continue

                    value = clean_value(
                        row[
                            column
                        ]
                    )

                    if value:

                        medications.extend(
                            parse_recommendation_value(
                                value
                            )
                        )


    medications = unique_values(
        medications
    )

    if medications:
        return medications


    return [
        "Medication information is provided for "
        "educational reference only. Consult a qualified "
        "healthcare professional before using any medicine."
    ]


# ============================================================
# DIET
# ============================================================

def get_diet(
    disease
):

    diet_values = []

    disease_key = normalize_text(
        disease
    )


    # --------------------------------------------------------
    # disease_recommendations.csv
    # --------------------------------------------------------

    if disease_key in recommendation_lookup:

        diet_values.extend(
            recommendation_lookup[
                disease_key
            ].get(
                "diet",
                []
            )
        )


    # --------------------------------------------------------
    # Dataset 3 through verified mapping
    # --------------------------------------------------------

    mapped_disease = get_dataset3_disease(
        disease
    )

    if mapped_disease:

        matched_rows = (
            find_all_disease_records(
                df_diet_3,
                mapped_disease
            )
        )

        if not matched_rows.empty:

            for _, row in (
                matched_rows.iterrows()
            ):

                for column in row.index:

                    if (
                        column
                        == "_normalized_disease"
                    ):
                        continue

                    value = clean_value(
                        row[
                            column
                        ]
                    )

                    if value:

                        diet_values.extend(
                            parse_recommendation_value(
                                value
                            )
                        )


    diet_values = unique_values(
        diet_values
    )

    if diet_values:
        return diet_values


    return [
        "Follow a balanced diet appropriate to your "
        "individual health needs and stay adequately hydrated."
    ]


# ============================================================
# WORKOUT
# ============================================================

def get_workout(
    disease
):

    workout_values = []

    disease_key = normalize_text(
        disease
    )


    # --------------------------------------------------------
    # disease_recommendations.csv
    # --------------------------------------------------------

    if disease_key in recommendation_lookup:

        workout_values.extend(
            recommendation_lookup[
                disease_key
            ].get(
                "workout",
                []
            )
        )


    # --------------------------------------------------------
    # Dataset 3 through verified mapping
    # --------------------------------------------------------

    mapped_disease = get_dataset3_disease(
        disease
    )

    if mapped_disease:

        matched_rows = (
            find_all_disease_records(
                df_workout_3,
                mapped_disease
            )
        )

        if not matched_rows.empty:

            for _, row in (
                matched_rows.iterrows()
            ):

                for column in row.index:

                    if (
                        column
                        == "_normalized_disease"
                    ):
                        continue

                    value = clean_value(
                        row[
                            column
                        ]
                    )

                    if value:

                        workout_values.extend(
                            parse_recommendation_value(
                                value
                            )
                        )


    workout_values = unique_values(
        workout_values
    )

    if workout_values:
        return workout_values


    return [
        "Choose physical activity according to your "
        "health condition and current fitness level. "
        "Consult a healthcare professional when needed."
    ]


# ============================================================
# DATA STATUS
# ============================================================

def get_data_status():

    return {

        "dataset_1_description":
            not df_description_1.empty,

        "dataset_1_precautions":
            not df_precaution_1.empty,

        "dataset_2_medical_data":
            not df_medical_2.empty,

        "dataset_3_description":
            not df_description_3.empty,

        "dataset_3_diet":
            not df_diet_3.empty,

        "dataset_3_medications":
            not df_medication_3.empty,

        "dataset_3_precautions":
            not df_precaution_3.empty,

        "dataset_3_workout":
            not df_workout_3.empty,

        "verified_mappings":
            len(mapping_lookup),

        "disease_recommendations":
            len(recommendation_lookup)
    }


# ============================================================
# INVALID DISEASE PROTECTION
# ============================================================

INVALID_DISEASE_VALUES = {
    "",
    "unknown",
    "none",
    "null",
    "insufficient",
    "insufficient symptoms",
    "no prediction",
    "unreliable",
    "unreliable prediction",
    "insufficient evidence"
}


def is_valid_disease(
    disease
):
    """
    Check whether a disease name is suitable for
    disease-specific recommendation lookup.
    """

    disease_key = normalize_text(
        disease
    )

    if not disease_key:
        return False

    if disease_key in INVALID_DISEASE_VALUES:
        return False

    return True


# ============================================================
# MAIN RECOMMENDATION FUNCTION
# ============================================================

def recommend(
    disease
):
    """
    Generate recommendation information for a valid
    predicted disease.

    IMPORTANT:

    This function does not perform disease prediction.

    It only retrieves recommendation information after
    disease_prediction.py has already determined that
    the prediction is reliable.
    """

    disease = clean_value(
        disease
    )


    # --------------------------------------------------------
    # Protect against invalid prediction states
    # --------------------------------------------------------

    if not is_valid_disease(
        disease
    ):

        return {

            "disease":
                "Insufficient Symptoms",

            "description":
                "No disease-specific recommendation is "
                "available because there is insufficient "
                "evidence for a reliable prediction.",

            "diet":
                [],

            "medication":
                [],

            "precautions":
                [],

            "workout":
                [],

            "data_status":
                {
                    "recommendations_available":
                        False,

                    "reason":
                        "No valid reliable disease prediction "
                        "was supplied to the recommendation engine."
                }
        }


    # --------------------------------------------------------
    # Retrieve disease-specific information
    # --------------------------------------------------------

    description = get_description(
        disease
    )

    medication = get_medications(
        disease
    )

    diet = get_diet(
        disease
    )

    precautions = get_precautions(
        disease
    )

    workout = get_workout(
        disease
    )


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {

        "disease":
            disease,

        "description":
            description,

        "diet":
            diet,

        "medication":
            medication,

        "precautions":
            precautions,

        "workout":
            workout,

        "data_status":
            get_data_status()
    }


# ============================================================
# TEST FUNCTION
# ============================================================

def test_recommendation(
    disease
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RECOMMENDATION TEST"
    )

    print(
        "=" * 70
    )

    print(
        f"Disease: {disease}"
    )


    result = recommend(
        disease
    )


    print(
        "\nDescription:"
    )

    print(
        result[
            "description"
        ]
    )


    print(
        "\nMedication:"
    )

    for item in result[
        "medication"
    ]:

        print(
            f"  • {item}"
        )


    print(
        "\nDiet:"
    )

    for item in result[
        "diet"
    ]:

        print(
            f"  • {item}"
        )


    print(
        "\nPrecautions:"
    )

    for item in result[
        "precautions"
    ]:

        print(
            f"  • {item}"
        )


    print(
        "\nWorkout:"
    )

    for item in result[
        "workout"
    ]:

        print(
            f"  • {item}"
        )


    print(
        "\nData Status:"
    )

    for key, value in (
        result[
            "data_status"
        ].items()
    ):

        print(
            f"  {key}: {value}"
        )


    print(
        "=" * 70
    )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "PERSONALIZED HEALTHCARE SYSTEM"
    )

    print(
        "Recommendation Engine"
    )

    print(
        "=" * 70
    )


    print(
        "\nLoaded Recommendation Records:",
        len(
            recommendation_lookup
        )
    )

    print(
        "Verified Dataset-3 Mappings:",
        len(
            mapping_lookup
        )
    )


    print(
        "\nData Status:"
    )

    for key, value in (
        get_data_status().items()
    ):

        print(
            f"  {key}: {value}"
        )


    # ========================================================
    # TEST 1
    # ========================================================

    test_recommendation(
        "Diabetes"
    )


    # ========================================================
    # TEST 2
    # ========================================================

    test_recommendation(
        "Hepatitis E"
    )


    # ========================================================
    # TEST 3
    # ========================================================

    test_recommendation(
        "Hypothyroidism"
    )


    # ========================================================
    # TEST 4
    # ========================================================
    #
    # Invalid prediction state.
    # This MUST NOT load disease-specific recommendations.
    # ========================================================

    test_recommendation(
        "Insufficient Symptoms"
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "Recommendation engine tests completed."
    )

    print(
        "=" * 70
    )