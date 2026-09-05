# ============================================================
# PERSONALIZED HEALTHCARE SYSTEM
# DISEASE PREDICTION MODULE
# ============================================================

import os
import joblib
import pandas as pd

from src.recommendation import recommend


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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "disease_model.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_names.pkl"
)

DATASET_1_PATH = os.path.join(
    DATASET_1_DIR,
    "dataset.csv"
)


# ============================================================
# CONFIDENCE CONFIGURATION
# ============================================================

HIGH_CONFIDENCE = 0.80
MODERATE_CONFIDENCE = 0.60


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

REQUIRED_FILES = [
    MODEL_PATH,
    ENCODER_PATH,
    FEATURE_PATH,
    DATASET_1_PATH
]

for file_path in REQUIRED_FILES:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            "Required project file not found:\n"
            f"{file_path}"
        )


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

try:

    model = joblib.load(
        MODEL_PATH
    )

    label_encoder = joblib.load(
        ENCODER_PATH
    )

    feature_names = joblib.load(
        FEATURE_PATH
    )

except Exception as error:

    raise RuntimeError(
        "Error while loading trained model files.\n"
        f"Details: {error}"
    )


# ============================================================
# FEATURE VALIDATION
# ============================================================

if not isinstance(
    feature_names,
    list
):

    feature_names = list(
        feature_names
    )


if len(feature_names) == 0:

    raise ValueError(
        "feature_names.pkl contains no features."
    )


# ============================================================
# MODEL VALIDATION
# ============================================================

if not hasattr(
    model,
    "predict"
):

    raise TypeError(
        "Loaded model does not support prediction."
    )


if not hasattr(
    model,
    "predict_proba"
):

    raise TypeError(
        "Loaded model does not support probability prediction."
    )


# ============================================================
# MODEL FEATURE COUNT VALIDATION
# ============================================================

MODEL_FEATURE_COUNT = getattr(
    model,
    "n_features_in_",
    None
)

if (
    MODEL_FEATURE_COUNT is not None
    and
    MODEL_FEATURE_COUNT != len(feature_names)
):

    raise ValueError(
        "Model feature count does not match "
        "feature_names.pkl.\n"
        f"Model expects: {MODEL_FEATURE_COUNT}\n"
        f"Feature file contains: {len(feature_names)}"
    )


# ============================================================
# MODEL CLASS VALIDATION
# ============================================================

MODEL_CLASS_COUNT = len(
    getattr(
        model,
        "classes_",
        []
    )
)

ENCODER_CLASS_COUNT = len(
    getattr(
        label_encoder,
        "classes_",
        []
    )
)


if MODEL_CLASS_COUNT == 0:

    raise ValueError(
        "The trained model contains no disease classes."
    )


if (
    MODEL_CLASS_COUNT != ENCODER_CLASS_COUNT
):

    raise ValueError(
        "Model classes and label encoder classes "
        "do not match.\n"
        f"Model classes: {MODEL_CLASS_COUNT}\n"
        f"Encoder classes: {ENCODER_CLASS_COUNT}"
    )


# ============================================================
# SYMPTOM NORMALIZATION
# ============================================================

def normalize_symptom(symptom):

    if symptom is None:
        return ""

    try:

        if pd.isna(symptom):
            return ""

    except Exception:
        pass

    symptom = str(
        symptom
    ).strip().lower()

    symptom = symptom.replace(
        "_",
        " "
    )

    symptom = symptom.replace(
        "-",
        " "
    )

    symptom = " ".join(
        symptom.split()
    )

    return symptom


# ============================================================
# SYMPTOM ALIASES
# ============================================================
#
# User-friendly symptom names which may not exactly match
# the feature names used by the trained model.
#
# IMPORTANT:
# These aliases do NOT change the model.
# They only convert user input into an existing model feature.
# ============================================================

SYMPTOM_ALIASES = {

    # Heart rate
    "fast heart rate":
        "increased heart rate",

    "rapid heart rate":
        "increased heart rate",

    "high heart rate":
        "increased heart rate",

    "racing heart":
        "increased heart rate",

    "heart beating fast":
        "increased heart rate",

    "heartbeat fast":
        "increased heart rate",

    "slow heart rate":
        "decreased heart rate",

    "low heart rate":
        "decreased heart rate",

    "slow heartbeat":
        "decreased heart rate",

    # Common variations
    "skin rash":
        "skin rash",

    "rash":
        "skin rash"
}


# ============================================================
# FEATURE LOOKUP
# ============================================================

def create_feature_lookup():

    lookup = {}

    for feature in feature_names:

        normalized_feature = normalize_symptom(
            feature
        )

        if normalized_feature:

            lookup[
                normalized_feature
            ] = feature

    return lookup


FEATURE_LOOKUP = create_feature_lookup()


# ============================================================
# APPLY SYMPTOM ALIAS
# ============================================================

def resolve_symptom_alias(symptom):

    normalized = normalize_symptom(
        symptom
    )

    if normalized in SYMPTOM_ALIASES:

        return normalize_symptom(
            SYMPTOM_ALIASES[
                normalized
            ]
        )

    return normalized


# ============================================================
# LOAD DATASET
# ============================================================

try:

    dataset_1 = pd.read_csv(
        DATASET_1_PATH
    )

except Exception as error:

    raise RuntimeError(
        "Unable to load Dataset 1.\n"
        f"Details: {error}"
    )


# ============================================================
# DATASET VALIDATION
# ============================================================

if dataset_1.empty:

    raise ValueError(
        "Dataset 1 is empty."
    )


if "Disease" not in dataset_1.columns:

    raise ValueError(
        "Dataset 1 must contain a 'Disease' column."
    )


# ============================================================
# NORMALIZE DISEASE COLUMN
# ============================================================

dataset_1 = dataset_1.copy()

dataset_1[
    "_normalized_disease"
] = dataset_1[
    "Disease"
].apply(
    normalize_symptom
)


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_input(symptoms):

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if symptoms is None:

        raise ValueError(
            "Symptoms cannot be None."
        )


    if isinstance(
        symptoms,
        str
    ):

        symptoms = [
            symptoms
        ]


    if not isinstance(
        symptoms,
        (list, tuple, set)
    ):

        raise TypeError(
            "Symptoms must be provided as "
            "a list, tuple, set or string."
        )


    # --------------------------------------------------------
    # Normalize symptoms
    # --------------------------------------------------------

    normalized_symptoms = []


    for symptom in symptoms:

        cleaned = normalize_symptom(
            symptom
        )

        if cleaned:

            normalized_symptoms.append(
                cleaned
            )


    # --------------------------------------------------------
    # Remove duplicate user inputs
    # --------------------------------------------------------

    normalized_symptoms = list(
        dict.fromkeys(
            normalized_symptoms
        )
    )


    if not normalized_symptoms:

        raise ValueError(
            "At least one valid symptom is required."
        )


    # ========================================================
    # CREATE EMPTY MODEL INPUT
    # ========================================================

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_names,
        dtype="int8"
    )


    # ========================================================
    # MATCH SYMPTOMS
    # ========================================================

    recognized_symptoms_list = []

    unknown_symptoms = []


    for original_symptom in normalized_symptoms:

        # ----------------------------------------------------
        # First try alias
        # ----------------------------------------------------

        resolved_symptom = resolve_symptom_alias(
            original_symptom
        )


        # ----------------------------------------------------
        # Match against trained feature lookup
        # ----------------------------------------------------

        if resolved_symptom in FEATURE_LOOKUP:

            original_feature = FEATURE_LOOKUP[
                resolved_symptom
            ]


            input_data.loc[
                0,
                original_feature
            ] = 1


            # Show the actual model feature
            recognized_symptoms_list.append(
                original_feature
            )


        else:

            unknown_symptoms.append(
                original_symptom
            )


    # --------------------------------------------------------
    # Remove duplicate recognized features
    # --------------------------------------------------------

    recognized_symptoms_list = list(
        dict.fromkeys(
            recognized_symptoms_list
        )
    )


    recognized_count = len(
        recognized_symptoms_list
    )


    # --------------------------------------------------------
    # At least one symptom must be recognized
    # --------------------------------------------------------

    if recognized_count == 0:

        raise ValueError(
            "None of the selected symptoms "
            "were recognized by the trained model."
        )


    # ========================================================
    # FEATURE ORDER VALIDATION
    # ========================================================

    if list(
        input_data.columns
    ) != list(
        feature_names
    ):

        raise ValueError(
            "Input feature order does not "
            "match the trained model."
        )


    return (
        input_data,
        unknown_symptoms,
        recognized_symptoms_list
    )


# ============================================================
# DECODE DISEASE
# ============================================================

def decode_disease(encoded_class):

    try:

        decoded = label_encoder.inverse_transform(
            [encoded_class]
        )[0]

        return str(
            decoded
        )

    except Exception:

        return str(
            encoded_class
        )


# ============================================================
# GENERATE ALL DISEASE PROBABILITIES
# ============================================================

def generate_probabilities(probabilities):

    class_probabilities = []


    for class_index, probability in enumerate(
        probabilities
    ):

        encoded_class = model.classes_[
            class_index
        ]


        disease_name = decode_disease(
            encoded_class
        )


        class_probabilities.append(
            {
                "disease":
                    disease_name,

                "probability":
                    float(
                        probability
                    )
            }
        )


    # Highest probability first

    class_probabilities.sort(
        key=lambda item:
        item["probability"],
        reverse=True
    )


    return class_probabilities


# ============================================================
# CONFIDENCE LEVEL
# ============================================================

def get_confidence_level(confidence):

    confidence = float(
        confidence
    )


    if confidence >= HIGH_CONFIDENCE:

        return "High"


    if confidence >= MODERATE_CONFIDENCE:

        return "Moderate"


    return "Low"


# ============================================================
# PREDICTION STATUS
# ============================================================

def get_prediction_status(confidence):

    confidence = float(
        confidence
    )


    if confidence >= HIGH_CONFIDENCE:

        return (
            "High-confidence machine learning prediction."
        )


    if confidence >= MODERATE_CONFIDENCE:

        return (
            "Moderate-confidence machine learning prediction."
        )


    return (
        "Low-confidence machine learning prediction. "
        "The model has returned its highest-probability "
        "disease, but the result should be interpreted "
        "with caution."
    )


# ============================================================
# DISEASE PREDICTION
# ============================================================

def predict_disease(symptoms):

    # ========================================================
    # PREPARE INPUT
    # ========================================================

    (
        input_data,
        unknown_symptoms,
        recognized_symptoms_list
    ) = prepare_input(
        symptoms
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(
        input_data
    )


    probabilities = model.predict_proba(
        input_data
    )[0]


    # ========================================================
    # FIND HIGHEST PROBABILITY
    # ========================================================

    predicted_index = int(
        probabilities.argmax()
    )


    predicted_class = model.classes_[
        predicted_index
    ]


    predicted_disease = decode_disease(
        predicted_class
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    # ========================================================
    # ALL PROBABILITIES
    # ========================================================

    class_probabilities = generate_probabilities(
        probabilities
    )


    # ========================================================
    # TOP 5
    # ========================================================

    top_5_predictions = class_probabilities[
        :5
    ]


    # ========================================================
    # RESULT INFORMATION
    # ========================================================

    recognized_symptoms = len(
        recognized_symptoms_list
    )


    reliable_prediction = True

    evidence_based = False

    prediction_source = (
        "Random Forest Model"
    )

    evidence_strength = ""

    evidence_support = 0

    disease = str(
        predicted_disease
    )


    prediction_status = get_prediction_status(
        confidence
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "disease":
            disease,

        "raw_predicted_disease":
            disease,

        "confidence":
            confidence,

        "confidence_level":
            get_confidence_level(
                confidence
            ),

        "recognized_symptoms":
            recognized_symptoms,

        "recognized_symptoms_list":
            recognized_symptoms_list,

        "unknown_symptoms":
            unknown_symptoms,

        "reliable_prediction":
            reliable_prediction,

        "prediction_status":
            prediction_status,

        "top_5_predictions":
            top_5_predictions,

        "prediction_source":
            prediction_source,

        "evidence_based":
            evidence_based,

        "evidence_strength":
            evidence_strength,

        "evidence_support":
            evidence_support
    }


# ============================================================
# PREDICTION + RECOMMENDATIONS
# ============================================================

def predict_with_recommendations(symptoms):

    prediction = predict_disease(
        symptoms
    )


    disease = prediction[
        "disease"
    ]


    # ========================================================
    # LOAD RECOMMENDATIONS
    # ========================================================

    try:

        recommendations = recommend(
            disease
        )


        if not isinstance(
            recommendations,
            dict
        ):

            recommendations = {

                "disease":
                    disease,

                "description":
                    "",

                "medication":
                    [],

                "diet":
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
                            "Recommendation engine "
                            "returned invalid data."
                    }
            }


    except Exception as error:

        recommendations = {

            "disease":
                disease,

            "description":
                "",

            "medication":
                [],

            "diet":
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
                        "Recommendation data is "
                        "not available for this disease.",

                    "error":
                        str(error)
                }
        }


    # ========================================================
    # MERGE RESULT
    # ========================================================

    result = {

        "disease":
            prediction[
                "disease"
            ],

        "raw_predicted_disease":
            prediction[
                "raw_predicted_disease"
            ],

        "confidence":
            prediction[
                "confidence"
            ],

        "confidence_level":
            prediction[
                "confidence_level"
            ],

        "recognized_symptoms":
            prediction[
                "recognized_symptoms"
            ],

        "recognized_symptoms_list":
            prediction[
                "recognized_symptoms_list"
            ],

        "unknown_symptoms":
            prediction[
                "unknown_symptoms"
            ],

        "reliable_prediction":
            prediction[
                "reliable_prediction"
            ],

        "prediction_status":
            prediction[
                "prediction_status"
            ],

        "top_5_predictions":
            prediction[
                "top_5_predictions"
            ],

        "prediction_source":
            prediction[
                "prediction_source"
            ],

        "evidence_based":
            prediction[
                "evidence_based"
            ],

        "evidence_strength":
            prediction[
                "evidence_strength"
            ],

        "evidence_support":
            prediction[
                "evidence_support"
            ],

        "recommendations":
            recommendations
    }


    return result


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info():

    return {

        "model_type":
            type(
                model
            ).__name__,

        "number_of_features":
            len(
                feature_names
            ),

        "number_of_classes":
            len(
                label_encoder.classes_
            ),

        "model_classes":
            list(
                label_encoder.classes_
            ),

        "confidence_threshold":
            "Informational only",

        "minimum_recognized_symptoms":
            1,

        "single_symptom_evidence":
            False,

        "recommendation_engine":
            True,

        "symptom_aliases":
            dict(
                SYMPTOM_ALIASES
            )
    }


# ============================================================
# DEBUG / TEST FUNCTION
# ============================================================

def run_test(
    test_name,
    symptoms
):

    print(
        "\n"
        + "=" * 75
    )

    print(
        f"TEST: {test_name}"
    )

    print(
        "=" * 75
    )


    print(
        "Input Symptoms:"
    )

    print(
        symptoms
    )


    try:

        result = predict_with_recommendations(
            symptoms
        )


        print(
            "\nPredicted Disease:"
        )

        print(
            result[
                "disease"
            ]
        )


        print(
            "\nConfidence:"
        )

        print(
            f"{result['confidence'] * 100:.2f}%"
        )


        print(
            "\nConfidence Level:"
        )

        print(
            result[
                "confidence_level"
            ]
        )


        print(
            "\nRecognized Symptoms:"
        )

        print(
            result[
                "recognized_symptoms"
            ]
        )


        print(
            "\nRecognized Symptom List:"
        )

        print(
            result[
                "recognized_symptoms_list"
            ]
        )


        print(
            "\nUnknown Symptoms:"
        )

        print(
            result[
                "unknown_symptoms"
            ]
        )


        print(
            "\nPrediction Source:"
        )

        print(
            result[
                "prediction_source"
            ]
        )


        print(
            "\nPrediction Status:"
        )

        print(
            result[
                "prediction_status"
            ]
        )


        print(
            "\nTop-5 Predictions:"
        )


        for item in result[
            "top_5_predictions"
        ]:

            print(
                f"  {item['disease']}: "
                f"{item['probability'] * 100:.2f}%"
            )


        recommendations = result[
            "recommendations"
        ]


        print(
            "\nRecommendation Status:"
        )

        print(
            recommendations.get(
                "data_status",
                {}
            )
        )


        print(
            "\nTEST PASSED"
        )


    except Exception as error:

        print(
            "\nTEST FAILED"
        )

        print(
            f"Error: {error}"
        )


    print(
        "=" * 75
    )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 75
    )

    print(
        "PERSONALIZED HEALTHCARE SYSTEM"
    )

    print(
        "Disease Prediction Module"
    )

    print(
        "=" * 75
    )


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    info = get_model_info()


    print(
        "\nModel Information:"
    )


    print(
        f"Model Type: "
        f"{info['model_type']}"
    )


    print(
        f"Number of Features: "
        f"{info['number_of_features']}"
    )


    print(
        f"Number of Disease Classes: "
        f"{info['number_of_classes']}"
    )


    # --------------------------------------------------------
    # DISPLAY ALL DISEASE CLASSES
    # --------------------------------------------------------

    print(
        "\nDisease Classes:"
    )


    for index, disease in enumerate(
        info["model_classes"],
        start=1
    ):

        print(
            f"{index:03d}. {disease}"
        )


    # ========================================================
    # TEST 1
    # ========================================================

    run_test(
        "Single Symptom - Abdominal Pain",
        [
            "abdominal pain"
        ]
    )


    # ========================================================
    # TEST 2
    # ========================================================

    run_test(
        "Single Symptom - Fast Heart Rate",
        [
            "fast heart rate"
        ]
    )


    # ========================================================
    # TEST 3
    # ========================================================

    run_test(
        "Multiple Symptoms",
        [
            "fatigue",
            "weight loss",
            "polyuria"
        ]
    )


    print(
        "\n"
        + "=" * 75
    )

    print(
        "Disease prediction module test completed."
    )

    print(
        "=" * 75
    )