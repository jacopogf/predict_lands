import warnings
warnings.filterwarnings("ignore")
import numpy as np
import joblib
import streamlit as st
import os
import gdown

# Load trained models
MODEL_PATH = ("trained_models.joblib")
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/file/d/1sqXtG4sI_MdXIzZ0dvcn3PiHv_dxallP/view?usp=share_link"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

models = joblib.load(MODEL_PATH)

def predict_lands(
        avg_mv, draw_spells, ramp_spells, has_companion=0,
        nonmythic_mdfc=0, mythic_mdfc=0, deck_size=100,
        models=models, model_choice=None
    ):
    draw_plus_ramp = draw_spells + ramp_spells
    total_mdfc = nonmythic_mdfc + mythic_mdfc

    base_vector = np.array([[
        avg_mv, draw_spells, ramp_spells, nonmythic_mdfc, mythic_mdfc,
        total_mdfc, draw_plus_ramp, has_companion
    ]])

    sorted_models = sorted(
        models.items(), key=lambda x: x[1]['model'].score(
            base_vector if x[1]['scaler'] is None else x[1]['scaler'].transform(base_vector),
            [0]  # dummy
        ), reverse=True
    )

    model_key = model_choice if model_choice else sorted_models[0][0]
    m = models[model_key]

    vec = m['scaler'].transform(base_vector) if m['scaler'] else base_vector
    pred = m['model'].predict(vec)[0] * (deck_size / 60)
    return model_key, round(pred)


# Streamlit interface
st.title("Recommended Lands Predictor")

avg_mv = st.number_input("Average Mana Value", value=2.37)
draw_spells = st.number_input("Cheap Card Draw", value=11)
ramp_spells = st.number_input("Cheap Mana Prod", value=10)
nonmythic_mdfc = st.number_input("Non-Mythic MDFCs", value=0)
mythic_mdfc = st.number_input("Mythic MDFCs", value=0)
has_companion = st.selectbox("Has Companion?", [0,1])
deck_size = st.number_input("Deck Size", value=100)

model_choice = st.selectbox("Model", ["Best","LR","RF","XGB"])
if model_choice == "Best":
    model_choice = None

if st.button("Predict"):
    model_used, lands_needed = predict_lands(
        avg_mv, draw_spells, ramp_spells, has_companion,
        nonmythic_mdfc, mythic_mdfc, deck_size,
        model_choice=model_choice
    )
    st.success(f"Model used: {model_used}, Recommended lands: {lands_needed}")
