import warnings
warnings.filterwarnings("ignore")
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from kagglehub import dataset_load, KaggleDatasetAdapter
import time

start_time = time.time()


# %%
# --------------------------
# Load dataset
# --------------------------
df = dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "frankkarsten/mtg-lands",
    path="All_MTG_decks_for_land_prediction.csv"
)

feature_map = {
    # NOTE: "XXX" needs to be replaced with actuale short names
    # "Deck hyperlink": "DHL",
    # "Total maindeck cards": "T"
    # "Lands (without MDFCs)": "L",
    "Average mana value (counting land/spell MDFCs as spells)": "AMV",
    "Nonmythic land/spell MDFCs": "NM-MDFCs",
    "Mythic land/spell MDFCs": "M-MDFCs",
    "Total land/spell MDFCs": "T-MDFCs",
    "Cheap card draw": "CCD",
    "Cheap mana prod": "CMP",
    "Sum of cheap card draw and cheap mana prod": "SCCDCMP",
    # "Date of event": "D",
    # "Format": "FMT",
    # "Companion (based on sideboard inclusion)": "XXX",
    "Companion present (1 or 0)": "COMP",
    # "Wins": "W",
    # "Losses": "L",
    # "Wins minus losses": "WML",
    # "Number of lands (incl partial MDFCs): "XXX"    
}
features = list(feature_map.keys())
X = df[features]
y = df["Number of lands (incl partial MDFCs)"]

random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)


# %%
# --------------------------
# Train models
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {}

def _fit_model(
        name, model, use_scaler=False,
        report_R2_RMSE=True,
        has_importance=True,
        report_feature_importance=True
    ):
    vec_train = X_train_scaled if use_scaler else X_train
    vec_test = X_test_scaled if use_scaler else X_test

    model.fit(vec_train, y_train)
    result = {
        'model': model,
        'scaler': scaler if use_scaler else None
    }

    if report_R2_RMSE:
        print(f"\n--- {name} ---")
        y_pred = model.predict(vec_test)
        print(
            f"R2: {round(r2_score(y_test, y_pred) * 100)}%, "
            f"RMSE: {round(np.sqrt(mean_squared_error(y_test, y_pred)))} lands"
        )

    k = 5 # folds

    X_full_scaled = scaler.transform(X) if use_scaler else X
    cv_r2 = cross_val_score(
        model, X_full_scaled, y, cv=k, scoring="r2"
    )
    cv_rmse = np.sqrt(
        -cross_val_score(
            model, X_full_scaled, y, cv=k,
            scoring="neg_mean_squared_error"
        )
    )
    result['CV_R2'] = cv_r2
    print(
        f"5-FCV R2: ({cv_r2.mean()*100:.0f} ± {cv_r2.std()*100:.0f})%, "
        f"RMSE: ({cv_rmse.mean():.1f} ± {cv_rmse.std():.1f}) lands"
    )
    
    if report_feature_importance and has_importance:
        importances = pd.Series(
            model.feature_importances_, index=features
        ).sort_values(ascending=False)
        print("Feature importance:")
        for f, imp in importances.items():
            print(f" {feature_map.get(f,f)}: {imp*100:.0f}%")

    models[name] = result


report_R2_RMSE, report_feature_importance = True, True

_fit_model(
    'LR', LinearRegression(), use_scaler=True,
    report_R2_RMSE=report_R2_RMSE,
    has_importance=False,
    report_feature_importance=report_feature_importance
)
_fit_model(
    'RF', RandomForestRegressor(n_estimators=200, random_state=random_state),
    use_scaler=False,
    report_R2_RMSE=report_R2_RMSE,
    has_importance=True,
    report_feature_importance=report_feature_importance
)
_fit_model(
    'XGB', xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=random_state
    ), use_scaler=False,
    report_R2_RMSE=report_R2_RMSE,
    has_importance=True,
    report_feature_importance=report_feature_importance
)


# %%
# --------------------------
# Save models
# --------------------------
file_name = "trained_models.joblib"
joblib.dump(models, file_name)
print(f"\nModels trained and saved to {file_name}")

end_time = time.time()
elapsed = end_time - start_time
print(f"\nElapsed time: {elapsed:.2f} s")


# %%