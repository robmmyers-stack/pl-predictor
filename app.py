import streamlit as st
import joblib
import pandas as pd

model = joblib.load("xgb_model.pkl")
le_result = joblib.load("le_result.pkl")
le = joblib.load("le_teams.pkl")
df = joblib.load("pl_data.pkl")

result_labels = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

def get_rolling_stats(df, team, season, match_idx, n=5):
    season_df = df[(df["Season"] == season) & (df.index < match_idx)]
    home = season_df[season_df["HomeTeam"] == team][["HomeGoals","AwayGoals","HST","AST"]].tail(n)
    away = season_df[season_df["AwayTeam"] == team][["AwayGoals","HomeGoals","AST","HST"]].tail(n)
    home.columns = ["scored","conceded","sot_for","sot_against"]
    away.columns = ["scored","conceded","sot_for","sot_against"]
    combined = pd.concat([home, away])
    if len(combined) == 0:
        return 1.5, 1.5, 4.0, 4.0
    return combined["scored"].mean(), combined["conceded"].mean(), combined["sot_for"].mean(), combined["sot_against"].mean()

def predict(home, away, home_injuries=0, away_injuries=0):
    last_season = df["Season"].iloc[0]
    last_idx = df.index.max()
    h = get_rolling_stats(df, home, last_season, last_idx)
    a = get_rolling_stats(df, away, last_season, last_idx)
    h_scored = h[0] * (1 - 0.05 * home_injuries)
    h_sot = h[2] * (1 - 0.05 * home_injuries)
    a_scored = a[0] * (1 - 0.05 * away_injuries)
    a_sot = a[2] * (1 - 0.05 * away_injuries)
    features = pd.DataFrame([[h_scored, h[1], h_sot, h[3], a_scored, a[1], a_sot, a[3]]],
                             columns=["H_Scored","H_Conceded","H_SOT","H_SOT_Against",
                                      "A_Scored","A_Conceded","A_SOT","A_SOT_Against"])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    return result_labels[le_result.classes_[pred]], dict(zip([result_labels[c] for c in le_result.classes_], proba))

st.title("Premier League Predictor")
teams = sorted(le.classes_)
tab1, tab2 = st.tabs(["Single Match", "Batch Fixtures"])

with tab1:
    col1, col2 = st.columns(2)
    home = col1.selectbox("Home Team", teams)
    away = col2.selectbox("Away Team", teams, index=1)
    home_injuries = st.number_input("Home Team Injuries", min_value=0, max_value=11, value=0)
    away_injuries = st.number_input("Away Team Injuries", min_value=0, max_value=11, value=0)
    if st.button("Predict", key="single"):
        if home == away:
            st.error("Please select two different teams.")
        else:
            result, probs = predict(home, away, home_injuries, away_injuries)
            st.subheader(f"Predicted: {result}")
            for label, prob in probs.items():
                st.progress(float(prob), text=f"{label}: {prob:.1%}")

with tab2:
    st.write("Enter fixtures one per line as: Home Team,Away Team,HomeInjuries,AwayInjuries")
    fixtures_input = st.text_area("Fixtures", "Arsenal,Chelsea,0,0\nLiverpool,Man City,1,2")
    if st.button("Predict All", key="batch"):
        lines = fixtures_input.strip().split("\n")
        results = []
        parlay_prob = 1.0
        for line in lines:
            try:
                parts = [x.strip() for x in line.split(",")]
                home, away = parts[0], parts[1]
                home_inj = int(parts[2]) if len(parts) > 2 else 0
                away_inj = int(parts[3]) if len(parts) > 3 else 0
                result, probs = predict(home, away, home_inj, away_inj)
                confidence = max(probs.values())
                parlay_prob *= confidence
                results.append({"Home": home, "Away": away, "H Inj": home_inj, "A Inj": away_inj, "Prediction": result, "Confidence": f"{confidence:.0%}"})
            except Exception as e:
                results.append({"Home": line, "Away": "", "H Inj": "", "A Inj": "", "Prediction": f"Error: {e}", "Confidence": ""})
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        st.markdown("---")
        st.metric(label=f"Parlay Probability (all {len(results)} correct)", value=f"{parlay_prob:.1%}")
