import streamlit as st
import pandas as pd
from scripts.predict import prediction

# read this season 
matches = pd.read_csv("pl_25_26.csv")

st.title("⚽ Premier League Match Predictor")

match_days = matches["round"].unique()
selected_day = st.selectbox("Select Matchday", match_days)

# the selected matchday 
day_matches = matches[matches["round"] == selected_day]

st.write(f"### Matches on {selected_day}")

# sorting the table by date and time
day_matches_sorted = day_matches.sort_values(by=["date", "time"]).reset_index(drop=True)

st.dataframe(day_matches_sorted[["date", "time", "team", "opponent", "venue"]])

# shows team vs opponent to select from
match_options = [f"{row['team']} vs {row['opponent']}" for _, row in day_matches_sorted.iterrows()]
selected_match = st.selectbox("Choose a Match to Predict", match_options)

# locate the row which is selected
match_row = day_matches_sorted.iloc[match_options.index(selected_match)]
team = str(match_row['team'])
opponent = str(match_row['opponent'])

# Predict button
if st.button("Predict Result"):

    # sends it to the prediction function
    prob, proba = prediction(match_row)

    outcome_mapping = {0: "Lose", 1: "Draw", 2: "Win"}
    st.write(f"### {team} to {(outcome_mapping[prob]).lower()} vs {opponent}")
    #st.write(f"#### Predicted Outcome: **{outcome_mapping[prob]}**")
    st.write(f"Probability Distribution: ")

    # clean way of displaying probabilities
    for cls, p in zip(outcome_mapping.values(), proba):
        st.write(f"  {cls}: {p:.2%}")