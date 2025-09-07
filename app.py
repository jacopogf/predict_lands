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


# ---- Streamlit interface ----
st.set_page_config(page_title="Lands Predictor", layout="centered")
st.title("Lands Predictor")
st.caption("Optimize lands count for your deck")

# --- Initialize session_state ---
default_values = {
    "avg_mv": 2.37,
    "draw_spells": 11,
    "ramp_spells": 10,
    "nonmythic_mdfc": 0,
    "mythic_mdfc": 0,
    "deck_size": 100
}

for key, val in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- Helper: slider + number_input ---
def slider_number_input(
        label, key, min_val, max_val, step, is_float=False, fmt="%.0f"
    ):
    col1, col2 = st.columns([3,1])
    with col1:
        val = st.slider(
            label, min_val, max_val,
            float(st.session_state[key]) if is_float else st.session_state[key],
            step=step
        )
        st.session_state[key] = float(val) if is_float else int(val)
    with col2:
        # key unico per number_input
        val2 = st.number_input(
            "", 
            min_val, max_val, 
            st.session_state[key], 
            step=step, 
            format=fmt if is_float else "%d",
            key=f"{key}_num"
        )
        st.session_state[key] = float(val2) if is_float else int(val2)


# --- Parametri ---
with st.expander("Insert deck params", expanded=True):
    slider_number_input(
        "Average Mana Value", "avg_mv", 1.0, 6.0, 0.01, is_float=True, fmt="%.2f"
    )
    slider_number_input("Cheap Card Draw", "draw_spells", 0, 50, 1)
    slider_number_input("Cheap Mana Prod", "ramp_spells", 0, 50, 1)
    slider_number_input("Non-Mythic MDFCs", "nonmythic_mdfc", 0, 20, 1)
    slider_number_input("Mythic MDFCs", "mythic_mdfc", 0, 20, 1)
    slider_number_input("Deck Size", "deck_size", 40, 200, 1)

# --- Companion ---
has_companion = st.radio("Has Companion?", [0, 1], index=0, horizontal=True)

# --- Model choice ---
model_choice = st.selectbox(
    "Choose Model", ["LR", "RF", "XGB"], index=0
)

# --- Predict button ---
if st.button("🔮 Predict"):
    model_used, lands_needed = predict_lands(
        st.session_state["avg_mv"],
        st.session_state["draw_spells"],
        st.session_state["ramp_spells"],
        has_companion,
        st.session_state["nonmythic_mdfc"],
        st.session_state["mythic_mdfc"],
        st.session_state["deck_size"],
        model_choice=model_choice
    )
    st.success(f"✅ Recommended Lands: **{lands_needed}**")


# "Best", 
# if model_choice == "Best":
#     model_choice = None