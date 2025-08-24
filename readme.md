# Premier League Match Predictor

A machine learning project to predict Premier League match outcomes. It combines **historical match data**, **team form**, and **advanced features** into a trained model, and provides an interactive **Streamlit dashboard** for predictions.  

## Features
- Predicts match result probabilities (Win / Draw / Loss)
- Uses historical data (form, goals, shots, xG, etc.)
- Streamlit UI to select fixtures and view predictions

## Data Collection  

All data was scraped from **FBref**  
- Match results, fixtures, and advanced stats (shots, xG, etc.) were extracted.  
- Data is cleaned and transformed before being passed into feature engineering. 

## Model Details  
- **Algorithm:** RandomForestClassifier (sklearn)  
- **Training Data:** FBref Premier League seasons (2022 → 2025)  
- **Target:** Match outcome (Win / Draw / Loss)  

## Performance  
The model was trained on **three full seasons (2022/23, 2023/24, 2024/25)** and then evaluated on the **2024/25 season**.  

- **Accuracy:** 51.2%  

## Streamlit App 
The web app allows you to:  
1. Select a **matchday** from the 2025/26 season.  
2. View fixtures for that day.  
3. Choose a match → get predicted probabilities for **Win / Draw / Loss**.  
