import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

matches = pd.read_csv("matches_data/my_matches.csv", index_col=0)

class MissingDict(dict):
    __missing__ = lambda self, key: key

def refactor_features(matches):

    matches['date'] = pd.to_datetime(matches['date'])


    matches["venue_code"] = matches["venue"].astype("category").cat.codes


    map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"} 
    mapping = MissingDict(**map_values)

    matches["new_team"] = matches["team"].map(mapping)
    matches = matches.drop(columns = "team")

    matches = matches.rename(columns={'new_team': 'team'})


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

    def rolling_averages(group, cols, new_cols):
        group = group.sort_values("date")
        rolling_stats = group[cols].rolling(10, min_periods=1, closed="left").mean()
        group[new_cols] = rolling_stats
        group[new_cols] = group[new_cols].fillna(0)
        return group
    
    cols = ["xg", "gf", "ga", "sh", "sot", "dist", "fk", "pk"]

    new_cols = [f"{c}_rolling" for c in cols]

    matches = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))

    matches = matches.droplevel('team')

    matches["result_code"] = matches["result"].map({
        "L":0,
        "W":3,
        "D":1
    })

    matches["team_form_5"] = matches.groupby(["season", "team"])["result_code"].transform(
        lambda x: x.rolling(5, min_periods=1, closed="left").mean()
    )

    matches["team_form_5"] = matches["team_form_5"].fillna(1)


    # matches["opp_result_code"] = matches["result"].map({
    #     "W":0,
    #     "D":1,
    #     "L":3
    # })

    # matches["opp_form_5"] = matches.groupby("opp_code")["opp_result_code"].transform(
    #     lambda x: x.rolling(5, min_periods=1, closed="left").mean()
    #     )

    # matches["opp_form_5"] = matches["opp_form_5"].fillna(0)

    # Create lookup using the team as the main team (not as opponent)
    team_form_lookup = matches[["season", "team", "date", "team_form_5"]].copy()

    # Merge opponent's form by matching opponent name with team name
    matches = matches.merge(
    team_form_lookup.rename(columns={"team": "opponent", "team_form_5": "opp_form_5"}),
    on=["season", "opponent", "date"],
    how="left"
    )


    matches["form_difference_5"] = matches["team_form_5"] - matches["opp_form_5"]
    matches["form_difference_5"] = matches["form_difference_5"].fillna(0)


    matches["total_team_points"] = (matches.groupby(["season", "team"])["result_code"].cumsum().shift(fill_value=0))
    matches[["total_team_points"]] = matches[["total_team_points"]].fillna(0)


    return matches

# training the model
matches = refactor_features(matches)

# training features list
predictors = ["venue_code", "opp_code", "hours", "day_code",
              "xg_rolling", "gf_rolling", "ga_rolling", "sh_rolling",
              "sot_rolling", "dist_rolling", "fk_rolling", "pk_rolling", 
              "team_form_5", "opp_form_5", "form_difference_5", "total_team_points"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=200, 
        min_samples_split=10, 
        class_weight="balanced", 
        max_depth=18, 
        random_state=1)
    )])

model.fit(matches[predictors], matches["target"])

matches.to_csv("test.csv")

joblib.dump(model, "PL_football_pipeline.pkl")


