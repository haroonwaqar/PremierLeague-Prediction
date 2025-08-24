import joblib
import pandas as pd
from scripts.train import refactor_features

# getting the model 
pipeline = joblib.load("PL_football_pipeline.pkl")

# read this season
new_matches = pd.read_csv("scripts/pl_25_26.csv")

# add the new features need for prediction 
new_matches = refactor_features(new_matches)

new_matches.to_csv("pl_25_26.csv")

new_matches = new_matches[["date","team","opponent","round", "venue_code",
        "opp_code", "hours", "day_code","xg_rolling", "gf_rolling", "ga_rolling",
        "sh_rolling", "sot_rolling", "dist_rolling", "fk_rolling", "pk_rolling", 
        "team_form_5", "opp_form_5", "form_difference_5", "total_team_points"]]

# Make prediction
def prediction(match_row):

    prediction_match = new_matches[
    (new_matches["date"] == match_row["date"]) &
    (new_matches["team"] == match_row["team"]) &
    (new_matches["opponent"] == match_row["opponent"]) &
    (new_matches["round"] == match_row["round"])
    ]

    # getting that exact row
    prediction_match = prediction_match.iloc[[0]]

    # dropping columns model don't need
    prediction_match = prediction_match.drop(columns=["date","team","opponent","round"])

    pred = pipeline.predict(prediction_match)
    proba = pipeline.predict_proba(prediction_match)

    return pred[0], proba[0]
