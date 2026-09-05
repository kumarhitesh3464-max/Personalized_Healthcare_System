import pandas as pd
import os

# ============================================================
# Project Path
# ============================================================

DATA_DIR = "data"

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "disease_recommendations.csv"
)

# ============================================================
# Disease-specific Recommendations
# Educational / Reference Guidance
# ============================================================

recommendations = [

    {
        "disease": "(vertigo) Paroymsal  Positional Vertigo",
        "diet": "Stay hydrated; eat regular balanced meals; include fruits, vegetables and whole grains; avoid excessive caffeine and alcohol if they trigger symptoms.",
        "workout": "Prefer gentle walking and balance exercises as tolerated; avoid sudden head movements and strenuous activity during dizziness."
    },

    {
        "disease": "AIDS",
        "diet": "Maintain a nutrient-dense balanced diet with adequate protein, calories, fruits and vegetables; follow individualized dietary advice from a healthcare professional.",
        "workout": "Light to moderate walking, stretching and resistance activity may be appropriate depending on overall health; exercise intensity should be individualized."
    },

    {
        "disease": "Acne",
        "diet": "Prefer a balanced diet rich in vegetables, fruits, whole grains and adequate water; limit highly refined and sugary foods if they appear to worsen symptoms.",
        "workout": "Regular moderate aerobic activity is generally suitable; shower and change sweaty clothes after exercise to help maintain skin hygiene."
    },

    {
        "disease": "Alcoholic hepatitis",
        "diet": "Avoid alcohol completely; maintain adequate nutrition with balanced meals and follow professional dietary guidance, particularly when liver disease is significant.",
        "workout": "Avoid strenuous exercise during active illness; gentle activity should only be considered according to medical advice and current condition."
    },

    {
        "disease": "Allergy",
        "diet": "Maintain a balanced diet and identify and avoid known food allergens; stay hydrated and choose fresh foods that are well tolerated.",
        "workout": "Walking, stretching and moderate aerobic activity can be appropriate; avoid known environmental triggers during exercise."
    },

    {
        "disease": "Arthritis",
        "diet": "Choose a balanced diet rich in vegetables, fruits, whole grains and healthy protein sources; maintain adequate hydration.",
        "workout": "Low-impact activities such as walking, swimming, cycling and gentle mobility exercises are generally preferred."
    },

    {
        "disease": "Bronchial Asthma",
        "diet": "Maintain a balanced diet with fruits, vegetables, whole grains and adequate hydration; avoid foods that are known personal triggers.",
        "workout": "Walking, cycling and other moderate aerobic activities can be suitable with appropriate asthma management; warm up gradually."
    },

    {
        "disease": "Cervical spondylosis",
        "diet": "Maintain a balanced diet with adequate protein, vegetables, fruits and hydration to support overall musculoskeletal health.",
        "workout": "Gentle neck mobility, posture exercises, stretching and low-impact walking may be appropriate; avoid painful or forceful neck movements."
    },

    {
        "disease": "Chicken pox",
        "diet": "Stay well hydrated and choose soft, nutritious foods; avoid foods that irritate mouth or throat lesions.",
        "workout": "Prioritize rest and avoid strenuous exercise while fever, fatigue or active symptoms are present."
    },

    {
        "disease": "Chronic cholestasis",
        "diet": "Follow a balanced diet with professional guidance; nutrient absorption and fat tolerance may require individualized dietary planning.",
        "workout": "Gentle walking and light activity may be suitable depending on energy levels and liver condition; avoid strenuous activity during significant symptoms."
    },

    {
        "disease": "Common Cold",
        "diet": "Stay hydrated and eat balanced meals including fruits, vegetables, soups and other easy-to-tolerate nutritious foods.",
        "workout": "Light walking or stretching may be suitable when symptoms are mild; rest when fever, significant fatigue or worsening symptoms are present."
    },

    {
        "disease": "Dengue",
        "diet": "Focus on adequate fluids and nutritious, easy-to-tolerate foods; follow medical advice regarding hydration and nutrition.",
        "workout": "Rest is important during active dengue; avoid strenuous exercise until recovery and medical clearance."
    },

    {
        "disease": "Diabetes",
        "diet": "Prefer high-fiber foods, vegetables, whole grains and balanced portions of protein and carbohydrates; limit highly sugary foods and drinks.",
        "workout": "Regular walking, cycling, resistance training and other moderate activity can support fitness and glucose management; individualize activity around treatment and glucose levels."
    },

    {
        "disease": "Dimorphic hemmorhoids(piles)",
        "diet": "Increase dietary fiber through vegetables, fruits, whole grains and legumes; drink adequate fluids and avoid excessive straining.",
        "workout": "Walking and other light-to-moderate activities are generally suitable; avoid activities that significantly increase discomfort."
    },

    {
        "disease": "Drug Reaction",
        "diet": "Avoid the suspected triggering medication or substance only under professional guidance; maintain hydration and eat foods that are well tolerated.",
        "workout": "Rest during significant symptoms; avoid strenuous exercise until the reaction has resolved and medical guidance is obtained."
    },

    {
        "disease": "Fungal infection",
        "diet": "Maintain a balanced diet with adequate protein, vegetables, fruits and hydration; avoid excessive added sugar as part of general healthy eating.",
        "workout": "Moderate activity such as walking can be appropriate; keep affected skin clean and dry and avoid activity that causes irritation."
    },

    {
        "disease": "GERD",
        "diet": "Prefer smaller meals and avoid personal trigger foods; limit very fatty, spicy or acidic foods when they worsen symptoms and avoid lying down soon after meals.",
        "workout": "Walking and moderate activity are generally suitable; avoid vigorous exercise immediately after eating and activities that clearly worsen reflux."
    },

    {
        "disease": "Gastroenteritis",
        "diet": "Prioritize fluids and oral rehydration when needed; choose easy-to-tolerate nutritious foods and gradually return to a normal balanced diet as symptoms improve.",
        "workout": "Rest during active vomiting, diarrhea or fever; resume light activity gradually after hydration and recovery."
    },

    {
        "disease": "Heart attack",
        "diet": "Follow a heart-healthy eating pattern emphasizing vegetables, fruits, whole grains, legumes and appropriate lean protein while limiting excess sodium and saturated fat.",
        "workout": "Exercise after a heart attack should follow a medically supervised cardiac rehabilitation plan; do not start strenuous exercise independently."
    },

    {
        "disease": "Hepatitis B",
        "diet": "Maintain balanced nutrition with adequate protein, fruits, vegetables and hydration; avoid alcohol and follow individualized liver-care advice.",
        "workout": "Light-to-moderate activity may be appropriate when stable; avoid strenuous exercise during significant fatigue or active illness."
    },

    {
        "disease": "Hepatitis C",
        "diet": "Choose balanced meals with vegetables, fruits, whole grains and adequate protein; avoid alcohol and maintain healthy hydration.",
        "workout": "Walking and moderate activity can be appropriate when tolerated; adjust intensity according to fatigue and medical guidance."
    },

    {
        "disease": "Hepatitis D",
        "diet": "Maintain balanced nutrition and adequate hydration; avoid alcohol and follow individualized dietary advice for liver health.",
        "workout": "Gentle walking and light activity may be appropriate when stable; avoid strenuous activity during active illness."
    },

    {
        "disease": "Hepatitis E",
        "diet": "Stay hydrated and consume balanced, easy-to-tolerate meals; avoid alcohol and follow professional dietary advice.",
        "workout": "Prioritize rest during acute illness; gradually return to light activity after recovery."
    },

    {
        "disease": "Hypertension",
        "diet": "Prefer vegetables, fruits, whole grains, legumes and lower-sodium foods; limit highly processed foods and excessive sodium.",
        "workout": "Regular moderate aerobic activity such as brisk walking, cycling or swimming is generally beneficial when medically appropriate."
    },

    {
        "disease": "Hyperthyroidism",
        "diet": "Maintain a balanced nutrient-dense diet with adequate calories and protein; individualized nutritional advice may be needed depending on severity.",
        "workout": "Use gentle-to-moderate activity according to symptoms; avoid strenuous exercise when experiencing significant palpitations, weakness or heat intolerance."
    },

    {
        "disease": "Hypoglycemia",
        "diet": "Eat regular balanced meals and snacks when appropriate; rapidly absorbable carbohydrate may be needed for an acute low-blood-glucose episode according to an individual's treatment plan.",
        "workout": "Exercise should be planned around glucose monitoring and treatment needs; avoid exercise when blood glucose is dangerously low."
    },

    {
        "disease": "Hypothyroidism",
        "diet": "Maintain a balanced diet containing adequate protein, vegetables, fruits, whole grains and appropriate sources of micronutrients.",
        "workout": "Walking, cycling, resistance training and flexibility exercises can be gradually introduced according to energy levels and fitness."
    },

    {
        "disease": "Impetigo",
        "diet": "Maintain a balanced diet and adequate hydration; choose nutritious foods that support general health.",
        "workout": "Light activity may be appropriate; avoid contact activities that could spread infection and keep affected areas clean."
    },

    {
        "disease": "Jaundice",
        "diet": "Stay hydrated and eat balanced, easy-to-tolerate meals; dietary needs should be individualized according to the underlying cause.",
        "workout": "Rest during significant fatigue or acute illness; resume gentle activity gradually as recovery occurs."
    },

    {
        "disease": "Malaria",
        "diet": "Maintain hydration and consume nutritious, easy-to-tolerate meals; follow medical treatment and professional nutrition advice.",
        "workout": "Rest during fever and acute illness; gradually return to light activity after recovery."
    },

    {
        "disease": "Migraine",
        "diet": "Maintain regular meals and hydration; identify and avoid personal food triggers and excessive caffeine when relevant.",
        "workout": "Gentle walking, stretching and moderate aerobic activity may help some people; avoid intense activity during an acute attack."
    },

    {
        "disease": "Osteoarthristis",
        "diet": "Choose a balanced diet rich in vegetables, fruits, whole grains and adequate protein while maintaining a healthy body weight.",
        "workout": "Low-impact walking, swimming, cycling, strengthening and mobility exercises are generally preferred."
    },

    {
        "disease": "Paralysis (brain hemorrhage)",
        "diet": "Follow individualized nutrition and swallowing guidance; emphasize balanced nutrition and adequate hydration as advised by the healthcare team.",
        "workout": "Exercise should be rehabilitation-based and supervised by appropriate professionals; avoid unsupervised strenuous activity."
    },

    {
        "disease": "Peptic ulcer diseae",
        "diet": "Eat balanced meals and avoid foods or drinks that clearly worsen symptoms; limit alcohol and avoid unnecessary irritants according to professional advice.",
        "workout": "Gentle walking may be suitable when stable; avoid strenuous activity during significant pain or acute symptoms."
    },

    {
        "disease": "Pneumonia",
        "diet": "Stay hydrated and consume nutritious, easy-to-tolerate meals with adequate protein and calories during recovery.",
        "workout": "Rest during acute illness and gradually resume light activity as breathing and energy improve; avoid strenuous exercise until recovered."
    },

    {
        "disease": "Psoriasis",
        "diet": "Maintain a balanced diet rich in vegetables, fruits, whole grains and healthy protein sources; limit excessive alcohol and highly processed foods.",
        "workout": "Regular moderate activity such as walking, cycling or swimming can support overall health; adapt activity to joint or skin symptoms."
    },

    {
        "disease": "Tuberculosis",
        "diet": "Maintain adequate calories and protein with balanced meals, fruits, vegetables and sufficient hydration; follow professional nutrition advice during treatment.",
        "workout": "Rest during active illness and gradually increase light activity as strength and respiratory status improve under medical guidance."
    },

    {
        "disease": "Typhoid",
        "diet": "Stay hydrated and choose easy-to-digest nutritious foods; gradually return to a normal balanced diet as recovery progresses.",
        "workout": "Rest during fever and acute illness; resume light activity gradually after recovery."
    },

    {
        "disease": "Urinary tract infection",
        "diet": "Drink adequate fluids unless medically restricted and maintain a balanced diet; avoid personal bladder irritants if they worsen symptoms.",
        "workout": "Light walking and normal daily activity may be suitable when symptoms are mild; rest if fever or significant illness is present."
    },

    {
        "disease": "Varicose veins",
        "diet": "Maintain a balanced diet rich in vegetables, fruits, whole grains and adequate hydration; maintain a healthy body weight.",
        "workout": "Walking, cycling, swimming and calf-strengthening movements can support circulation; avoid prolonged immobility."
    },

    {
        "disease": "hepatitis A",
        "diet": "Stay hydrated and consume balanced, nutritious meals; avoid alcohol and follow professional advice if appetite or liver function is affected.",
        "workout": "Rest during acute illness and gradually return to light activity as energy and symptoms improve."
    }
]


# ============================================================
# Create DataFrame
# ============================================================

df_recommendations = pd.DataFrame(
    recommendations
)


# ============================================================
# Validation
# ============================================================

if len(df_recommendations) != 41:
    raise ValueError(
        f"Expected 41 diseases, found {len(df_recommendations)}."
    )


if df_recommendations["disease"].duplicated().any():
    raise ValueError(
        "Duplicate disease names found."
    )


if df_recommendations[
    ["disease", "diet", "workout"]
].isnull().any().any():
    raise ValueError(
        "Missing recommendation data found."
    )


# ============================================================
# Save CSV
# ============================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

df_recommendations.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)


# ============================================================
# Final Verification
# ============================================================

print("Recommendation dataset created successfully!")
print("Total diseases:", len(df_recommendations))
print("Saved at:", OUTPUT_PATH)

print("\nColumns:")
print(df_recommendations.columns.tolist())

print("\nFirst 5 records:")
print(
    df_recommendations.head()
)