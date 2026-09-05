import streamlit as st
import pandas as pd

from src.disease_prediction import (
    predict_with_recommendations,
    feature_names,
    model,
    get_model_info
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Personalized Healthcare System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 25px;
    }

    .disclaimer {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0b84c;
        background-color: #fff8e6;
        color: #333333;
    }

    .disclaimer h4 {
        color: #222222;
        margin-top: 0;
    }

    .disclaimer p {
        color: #333333;
        line-height: 1.6;
    }

    .result-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #d9d9d9;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🏥 Personalized Healthcare & Medicine Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based symptom analysis with '
    'educational healthcare recommendations.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("ℹ️ System Information")

    st.write(
        "This application uses a trained Random Forest "
        "machine learning model for symptom-based "
        "disease prediction."
    )

    st.divider()

    model_info = get_model_info()

    st.metric(
        "Model Features",
        model_info["number_of_features"]
    )

    st.metric(
        "Disease Classes",
        model_info["number_of_classes"]
    )

    st.metric(
        "Model Type",
        model_info["model_type"]
    )

    st.divider()

    st.caption(
        "Educational project only. "
        "Not a medical diagnosis or prescription."
    )


# ============================================================
# SYMPTOM SELECTION
# ============================================================

st.subheader("🩺 Select Your Symptoms")

st.write(
    "Search and select the symptoms you are currently "
    "experiencing."
)


# ============================================================
# CREATE SYMPTOM OPTIONS
# ============================================================

symptom_options = {}

for feature in feature_names:

    display_name = (
        str(feature)
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )

    symptom_options[display_name] = feature


# ============================================================
# MULTISELECT
# ============================================================

selected_symptom_names = st.multiselect(
    "Choose symptoms:",
    options=sorted(symptom_options.keys()),
    placeholder="Type here to search symptoms..."
)


# ============================================================
# CONVERT DISPLAY NAMES TO MODEL FEATURES
# ============================================================

symptoms = [
    symptom_options[name]
    for name in selected_symptom_names
]


# ============================================================
# SELECTED SYMPTOMS
# ============================================================

if selected_symptom_names:

    st.markdown(
        "### ✅ Selected Symptoms"
    )

    st.write(
        " • ".join(selected_symptom_names)
    )

    st.caption(
        f"{len(selected_symptom_names)} "
        "symptom(s) selected"
    )

else:

    st.info(
        "Please select one or more symptoms."
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Disease",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if not symptoms:

        st.warning(
            "⚠️ Please select at least one symptom "
            "before prediction."
        )

    else:

        try:

            with st.spinner(
                "Analyzing symptoms..."
            ):

                result = predict_with_recommendations(
                    symptoms
                )


            # ==================================================
            # RESULT VARIABLES
            # ==================================================

            disease = result.get(
                "disease",
                "Insufficient Symptoms"
            )

            confidence = float(
                result.get(
                    "confidence",
                    0.0
                )
            )

            confidence_percentage = (
                confidence * 100
            )

            reliable_prediction = bool(
                result.get(
                    "reliable_prediction",
                    False
                )
            )

            prediction_status = result.get(
                "prediction_status",
                ""
            )

            prediction_source = result.get(
                "prediction_source",
                "Machine Learning Model"
            )

            evidence_based = bool(
                result.get(
                    "evidence_based",
                    False
                )
            )

            evidence_strength = result.get(
                "evidence_strength",
                ""
            )

            evidence_support = result.get(
                "evidence_support",
                0
            )

            recognized_symptoms = int(
                result.get(
                    "recognized_symptoms",
                    0
                )
            )

            recognized_symptoms_list = result.get(
                "recognized_symptoms_list",
                []
            )

            unknown_symptoms = result.get(
                "unknown_symptoms",
                []
            )

            recommendations = result.get(
                "recommendations",
                {}
            )


            # ==================================================
            # PREDICTION STATUS
            # ==================================================

            st.success(
                "Prediction analysis completed."
            )


            # ==================================================
            # PREDICTION RESULT
            # ==================================================

            st.subheader(
                "🧠 Prediction Result"
            )


            # ==================================================
            # RELIABLE RESULT
            # ==================================================

            if reliable_prediction:

                # ----------------------------------------------
                # Disease name is shown ONLY when prediction
                # is considered reliable.
                # ----------------------------------------------

                st.success(
                    f"Predicted Disease: "
                    f"{str(disease).title()}"
                )


                # ----------------------------------------------
                # EVIDENCE-BASED RESULT
                # ----------------------------------------------

                if evidence_based:

                    st.info(
                        "📚 Evidence-Based Result: "
                        "The selected symptom has a strong "
                        "association with this disease in the "
                        "project's verified training data."
                    )


                    if evidence_strength:

                        st.caption(
                            f"Evidence strength: "
                            f"{evidence_strength}"
                        )


                    if evidence_support:

                        st.caption(
                            f"Supporting training records: "
                            f"{evidence_support}"
                        )


                # ----------------------------------------------
                # NORMAL ML RESULT
                # ----------------------------------------------

                else:

                    st.caption(
                        f"Prediction source: "
                        f"{prediction_source}"
                    )


            # ==================================================
            # UNRELIABLE RESULT
            # ==================================================

            else:

                st.warning(
                    "⚠️ Insufficient evidence for "
                    "a reliable disease prediction."
                )


                st.info(
                    "The selected symptom(s) do not provide "
                    "sufficient evidence for a reliable disease "
                    "prediction. Please select additional "
                    "relevant symptoms."
                )


                st.caption(
                    "No disease is being presented as a "
                    "confirmed prediction."
                )


                if prediction_source:

                    st.caption(
                        f"Prediction source: "
                        f"{prediction_source}"
                    )


            # ==================================================
            # PREDICTION SUMMARY
            # ==================================================

            st.divider()

            st.subheader(
                "📌 Prediction Summary"
            )


            (
                summary_col1,
                summary_col2,
                summary_col3,
                summary_col4
            ) = st.columns(4)


            with summary_col1:

                st.metric(
                    "🩺 Selected Symptoms",
                    len(symptoms)
                )


            with summary_col2:

                st.metric(
                    "✅ Recognized Symptoms",
                    recognized_symptoms
                )


            with summary_col3:

                if reliable_prediction:

                    display_disease = (
                        str(disease).title()
                    )

                else:

                    display_disease = (
                        "Insufficient"
                    )


                st.metric(
                    "🎯 Prediction",
                    display_disease
                )


            with summary_col4:

                st.metric(
                    "📊 Confidence",
                    f"{confidence_percentage:.2f}%"
                )


            # ==================================================
            # PREDICTION CONFIDENCE
            # ==================================================

            st.subheader(
                "📊 Prediction Confidence"
            )


            confidence_col1, confidence_col2 = (
                st.columns(2)
            )


            with confidence_col1:

                st.metric(
                    "Model Confidence",
                    f"{confidence_percentage:.2f}%"
                )


            with confidence_col2:

                if evidence_based:

                    st.success(
                        "Evidence-based"
                    )

                elif reliable_prediction:

                    if confidence >= 0.80:

                        st.success(
                            "High confidence"
                        )

                    elif confidence >= 0.60:

                        st.warning(
                            "Moderate confidence"
                        )

                    elif confidence >= 0.50:

                        st.warning(
                            "Low-Moderate confidence"
                        )

                    else:

                        st.warning(
                            "Low confidence"
                        )

                else:

                    st.error(
                        "Insufficient confidence"
                    )


            st.progress(
                min(
                    max(
                        confidence,
                        0.0
                    ),
                    1.0
                )
            )


            if prediction_status:

                st.caption(
                    prediction_status
                )


            # ==================================================
            # TOP-5 DISEASE PROBABILITIES
            # ==================================================
            #
            # IMPORTANT:
            #
            # If prediction is unreliable, the top-5 disease
            # names are NOT displayed.
            #
            # This prevents misleading output such as:
            #
            # Allergy 13%
            #
            # when the system itself says the evidence is
            # insufficient.
            # ==================================================

            st.divider()

            st.subheader(
                "📈 Disease Probability Analysis"
            )


            if reliable_prediction:

                top_5_predictions = result.get(
                    "top_5_predictions",
                    []
                )


                if top_5_predictions:

                    probability_data = pd.DataFrame(
                        top_5_predictions
                    )


                    if (
                        "disease" in probability_data.columns
                        and
                        "probability" in probability_data.columns
                    ):

                        probability_data["Disease"] = (
                            probability_data["disease"]
                            .astype(str)
                            .str.title()
                        )


                        probability_data["Probability (%)"] = (
                            probability_data["probability"]
                            .astype(float)
                            * 100
                        )


                        chart_data = (
                            probability_data[
                                [
                                    "Disease",
                                    "Probability (%)"
                                ]
                            ]
                            .set_index(
                                "Disease"
                            )
                        )


                        st.bar_chart(
                            chart_data
                        )


                        display_table = (
                            probability_data[
                                [
                                    "Disease",
                                    "Probability (%)"
                                ]
                            ]
                            .copy()
                        )


                        display_table[
                            "Probability (%)"
                        ] = (
                            display_table[
                                "Probability (%)"
                            ]
                            .round(2)
                        )


                        st.dataframe(
                            display_table,
                            use_container_width=True,
                            hide_index=True
                        )


                        st.caption(
                            "These values represent the model's "
                            "estimated probability distribution "
                            "across the five highest-ranked diseases."
                        )

                    else:

                        st.info(
                            "Disease probability data is "
                            "not available in the expected format."
                        )

                else:

                    st.info(
                        "Top disease probability data "
                        "is not available."
                    )


            else:

                st.info(
                    "Disease probability details are hidden "
                    "because the current symptom information "
                    "does not provide sufficient evidence for "
                    "a reliable prediction."
                )


            # ==================================================
            # SYMPTOM ANALYSIS
            # ==================================================

            st.divider()

            st.subheader(
                "🔎 Symptom Analysis"
            )


            total_symptoms = len(
                symptoms
            )

            unrecognized_count = len(
                unknown_symptoms
            )


            (
                analysis_col1,
                analysis_col2,
                analysis_col3
            ) = st.columns(3)


            with analysis_col1:

                st.metric(
                    "Selected",
                    total_symptoms
                )


            with analysis_col2:

                st.metric(
                    "Recognized",
                    recognized_symptoms
                )


            with analysis_col3:

                st.metric(
                    "Unrecognized",
                    unrecognized_count
                )


            # ==================================================
            # RECOGNITION RATE
            # ==================================================

            if total_symptoms > 0:

                recognition_rate = (
                    recognized_symptoms
                    /
                    total_symptoms
                )


                st.write(
                    f"**Model Recognition Rate:** "
                    f"{recognition_rate * 100:.1f}%"
                )


                st.progress(
                    min(
                        max(
                            recognition_rate,
                            0.0
                        ),
                        1.0
                    )
                )


            # ==================================================
            # RECOGNIZED SYMPTOMS
            # ==================================================

            if recognized_symptoms_list:

                with st.expander(
                    "✅ Recognized Symptoms",
                    expanded=False
                ):

                    for symptom in (
                        recognized_symptoms_list
                    ):

                        st.write(
                            f"• "
                            f"{str(symptom).replace('_', ' ').title()}"
                        )


            # ==================================================
            # UNKNOWN SYMPTOMS
            # ==================================================

            if unknown_symptoms:

                st.warning(
                    "Some entered symptoms were not "
                    "recognized by the trained model."
                )


                for symptom in unknown_symptoms:

                    st.write(
                        f"⚠️ "
                        f"{str(symptom).replace('_', ' ').title()}"
                    )

            else:

                st.success(
                    "✅ All selected symptoms were "
                    "recognized by the trained model."
                )


            # ==================================================
            # MODEL EXPLAINABILITY
            # ==================================================

            st.divider()

            st.subheader(
                "🧠 Model Explainability"
            )


            st.write(
                "The chart below shows the symptoms with "
                "the highest overall feature importance "
                "in the trained Random Forest model."
            )


            if hasattr(
                model,
                "feature_importances_"
            ):

                importance_data = pd.DataFrame(
                    {
                        "Symptom":
                            feature_names,

                        "Importance":
                            model.feature_importances_
                    }
                )


                importance_data = (
                    importance_data
                    .sort_values(
                        by="Importance",
                        ascending=False
                    )
                    .head(15)
                )


                importance_data["Symptom"] = (
                    importance_data["Symptom"]
                    .astype(str)
                    .str.replace(
                        "_",
                        " ",
                        regex=False
                    )
                    .str.replace(
                        "-",
                        " ",
                        regex=False
                    )
                    .str.title()
                )


                importance_chart = (
                    importance_data
                    .set_index(
                        "Symptom"
                    )
                )


                st.bar_chart(
                    importance_chart,
                    y="Importance"
                )


                st.caption(
                    "Feature importance represents the "
                    "relative importance of each symptom "
                    "across the Random Forest decision trees. "
                    "It does not establish medical causation."
                )


            else:

                st.info(
                    "Feature importance is not available "
                    "for the current model."
                )


            # ==================================================
            # HEALTHCARE INFORMATION
            # ==================================================

            st.divider()

            st.subheader(
                "📋 Healthcare Information"
            )


            # ==================================================
            # RECOMMENDATION AVAILABILITY
            # ==================================================

            has_recommendations = False


            if isinstance(
                recommendations,
                dict
            ):

                for key in [
                    "description",
                    "medication",
                    "diet",
                    "precautions",
                    "workout"
                ]:

                    value = recommendations.get(
                        key
                    )


                    if value:

                        has_recommendations = True

                        break


            # ==================================================
            # SHOW RECOMMENDATIONS
            # ==================================================
            #
            # Recommendations are displayed only when:
            #
            # 1. Prediction is reliable
            # 2. Recommendation data exists
            #
            # Therefore a disease for which recommendation
            # data has not been prepared will NOT receive
            # fake / incomplete healthcare information.
            # ==================================================

            if (
                reliable_prediction
                and
                has_recommendations
            ):

                # ----------------------------------------------
                # DESCRIPTION
                # ----------------------------------------------

                with st.expander(
                    "📖 Disease Description",
                    expanded=True
                ):

                    description = recommendations.get(
                        "description",
                        "No description available."
                    )


                    st.write(
                        description
                    )


                # ----------------------------------------------
                # MEDICATION
                # ----------------------------------------------

                with st.expander(
                    "💊 Medication Reference",
                    expanded=True
                ):

                    st.warning(
                        "Medication information is provided "
                        "for educational/reference purposes only. "
                        "Do not self-medicate. Consult a qualified "
                        "healthcare professional."
                    )


                    medication_data = (
                        recommendations.get(
                            "medication",
                            []
                        )
                    )


                    if medication_data:

                        for medicine in (
                            medication_data
                        ):

                            st.write(
                                f"• {medicine}"
                            )

                    else:

                        st.write(
                            "No medication reference data "
                            "is available."
                        )


                # ----------------------------------------------
                # DIET
                # ----------------------------------------------

                with st.expander(
                    "🥗 Disease-Specific Diet",
                    expanded=True
                ):

                    diet_data = (
                        recommendations.get(
                            "diet",
                            []
                        )
                    )


                    if diet_data:

                        for diet in diet_data:

                            st.write(
                                f"• {diet}"
                            )

                    else:

                        st.write(
                            "No diet recommendation data "
                            "is available."
                        )


                # ----------------------------------------------
                # PRECAUTIONS
                # ----------------------------------------------

                with st.expander(
                    "⚠️ Precautions",
                    expanded=True
                ):

                    precaution_data = (
                        recommendations.get(
                            "precautions",
                            []
                        )
                    )


                    if precaution_data:

                        for precaution in (
                            precaution_data
                        ):

                            st.write(
                                f"• {precaution}"
                            )

                    else:

                        st.write(
                            "No precaution information "
                            "is available."
                        )


                # ----------------------------------------------
                # WORKOUT
                # ----------------------------------------------

                with st.expander(
                    "🏃 Disease-Specific Workout",
                    expanded=True
                ):

                    workout_data = (
                        recommendations.get(
                            "workout",
                            []
                        )
                    )


                    if workout_data:

                        for workout in (
                            workout_data
                        ):

                            st.write(
                                f"• {workout}"
                            )

                    else:

                        st.write(
                            "No workout recommendation data "
                            "is available."
                        )


                # ----------------------------------------------
                # RECOMMENDATION DATA STATUS
                # ----------------------------------------------

                with st.expander(
                    "🔧 Recommendation Data Status"
                ):

                    status = recommendations.get(
                        "data_status",
                        {}
                    )


                    if status:

                        for key, value in (
                            status.items()
                        ):

                            if isinstance(
                                value,
                                bool
                            ):

                                icon = (
                                    "✅"
                                    if value
                                    else "❌"
                                )


                                st.write(
                                    f"{icon} {key}"
                                )

                            else:

                                st.write(
                                    f"• {key}: {value}"
                                )

                    else:

                        st.info(
                            "Recommendation data status "
                            "is not available."
                        )


            # ==================================================
            # RELIABLE PREDICTION BUT NO RECOMMENDATIONS
            # ==================================================

            elif reliable_prediction:

                st.info(
                    "A reliable disease prediction was obtained, "
                    "but healthcare recommendation information "
                    "for this disease is not currently available "
                    "in the project database."
                )


                st.caption(
                    "No unsupported medication, diet, workout "
                    "or precaution information has been generated."
                )


            # ==================================================
            # INSUFFICIENT EVIDENCE
            # ==================================================

            else:

                st.warning(
                    "Healthcare recommendations are not displayed "
                    "because the current symptom information does "
                    "not provide sufficient evidence for a reliable "
                    "prediction."
                )


                st.info(
                    "Please provide additional relevant symptoms "
                    "or consult a qualified healthcare professional."
                )


            # ==================================================
            # MEDICAL DISCLAIMER
            # ==================================================

            st.divider()

            st.markdown(
                """
                <div class="disclaimer">

                <h4>⚠️ Medical Disclaimer</h4>

                <p>
                This application is an educational
                machine-learning project. Disease predictions
                are generated from the trained model and should
                not be considered a medical diagnosis.
                </p>

                <p>
                Medication, diet, workout and precaution
                information is provided for educational/reference
                purposes only.
                </p>

                <p>
                Do not self-medicate or change treatment based
                solely on this application. Always consult a
                qualified healthcare professional for personalized
                medical advice.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ======================================================
        # ERROR HANDLING
        # ======================================================

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )

            st.exception(
                error
            )