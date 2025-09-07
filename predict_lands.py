import warnings
warnings.filterwarnings("ignore")
import numpy as np
import joblib
import os
import gdown


# %%
# Load trained models
models = joblib.load("trained_models.joblib")
if not os.path.exists(models):
    url = "https://drive.google.com/file/d/1sqXtG4sI_MdXIzZ0dvcn3PiHv_dxallP/view?usp=share_link"
    gdown.download(url, models, quiet=False)


# %%
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


def interactive_prediction():
    print("\nEnter the following values for your deck:")
    print(
        "[Average Mana Value, Cheap Card Draw, Cheap Mana Prod, "
        "Non-Mythic MDFCs, Mythic MDFCs, Companion]"
    )
    print("Example: 2.37 11 10 0 0 0\n")

    while True:
        user_input = input("Values or 'exit': ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting.")
            break

        try:
            values = list(map(float, user_input.strip().split()))
            if len(values) != 6:
                print("Error: insert exactly 6 values.")
                continue

            deck_size_input = input("Decksize (default 100): ").strip()
            deck_size = int(deck_size_input) if deck_size_input else 100

            model_choice = input(
                "Choose model (LR, RF, XGB) or leave empty for best: "
            ).strip().upper()
            if model_choice not in ['LR','RF','XGB','']:
                print("Invalid model, best will be chosen.")
                model_choice = None
            elif model_choice == '':
                model_choice = None

            model_used, lands_needed = predict_lands(
                avg_mv=values[0], draw_spells=values[1], ramp_spells=values[2],
                has_companion=int(values[5]),
                nonmythic_mdfc=values[3], mythic_mdfc=values[4],
                deck_size=deck_size,
                model_choice=model_choice
            )
            print(
                f"\n---- Recommended Lands ----\n"
                f"Model used: {model_used}, Lands: {lands_needed}\n"
            )

        except Exception as e:
            print(f"Input error: {e}. Retry.")


# %%
if __name__ == "__main__":
    interactive_prediction()


# %%