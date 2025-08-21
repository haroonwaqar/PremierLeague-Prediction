import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

matches = pd.read_csv("my_matches.csv", index_col=0)

def refactor_features(matches):

    matches['date'] = pd.to_datetime(matches['date'])


    matches["venue_code"] = matches["venue"].astype("category").cat.codes


    matches["opp_code"] = matches["opponent"].astype("category").cat.codes


    matches["hours"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")


    matches["day_code"] = matches["date"].dt.dayofweek


    matches["target"] = matches["result"].map({
        "L": 0,
        "D": 1, 
        "W": 2
    })


    #    "L" is 0
    #    "D" is 1
    #    "W" is 2

    matches = matches.sort_values(["season", "team", "date"]).reset_index(drop=True)


    matches["result_code"] = matches["result"].map({
        "L":0,
        "W":3,
        "D":1
    })


    matches["team_form_5"] = matches.groupby(["season", "team"])["result_code"].transform(
        lambda x: x.shift().rolling(5, min_periods=1, closed="left").mean()
        )

    matches["team_form_5"] = matches["team_form_5"].fillna(1)



    matches["avg_gf_5"] = matches.groupby("team")["gf"].transform(
        lambda x: x.shift().rolling(5, min_periods=1, closed="left").mean()
        )

    matches["avg_ga_5"] = matches.groupby("team")["ga"].transform(
        lambda x: x.shift().rolling(5, min_periods=1, closed="left").mean()
        )

    matches["avg_gd_5"] = matches["avg_gf_5"] - matches["avg_ga_5"]

    matches[["avg_gd_5","avg_ga_5","avg_gf_5"]] = matches[["avg_gd_5","avg_ga_5","avg_gf_5"]].fillna(0)


    matches["roll_xg_10"] = matches.groupby("team")["xg"].transform(
        lambda x: x.shift().rolling(10, min_periods=1, closed="left").mean()
        )

    matches["roll_xg_10"] = matches["roll_xg_10"].fillna(0)


    matches["opp_result_code"] = matches["result"].map({
        "W":0,
        "D":1,
        "L":3
    })


    matches["opp_form_5"] = matches.groupby("opp_code")["opp_result_code"].transform(
        lambda x: x.rolling(5, min_periods=1, closed="left").mean()
        )

    matches["opp_form_5"] = matches["opp_form_5"].fillna(1)


    matches["form_difference_5"] = matches["team_form_5"] - matches["opp_form_5"]


    matches["avg_sh_5"] = matches.groupby(["season","team"])["sh"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
        )


    matches["avg_sot_5"] = matches.groupby(["season","team"])["sot"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
        )

    matches[["avg_sh_5","avg_sot_5"]] = matches[["avg_sh_5","avg_sot_5"]].fillna(0)


    matches["total_team_points"] = (matches.groupby(["season", "team"])["result_code"].cumsum().shift(fill_value=0))
    matches[["total_team_points"]] = matches[["total_team_points"]].fillna(0)


    matches = matches.drop(columns=["attendance", "notes" ])

    return matches

# Training the model ---

matches = refactor_features(matches)

predictors = ["venue_code", "opp_code", "hours", "day_code", "team_form_5", 
              "avg_gf_5", "avg_ga_5", "avg_gd_5", "roll_xg_10", "opp_form_5", 
              "form_difference_5", "avg_sh_5", "avg_sot_5", "total_team_points"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=200, 
            min_samples_split=10, class_weight="balanced", 
            max_depth=12, random_state=1)
    )])

model.fit(matches[predictors], matches["target"])

joblib.dump(model, "PL_football_pipeline.pkl")


