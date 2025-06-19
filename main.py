# main.py (updated to support team1_year and team2_year)

from fastapi import FastAPI, HTTPException
from supabase import create_client
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

model = xgb.Booster()
model.load_model("march_madnessxg_sq.model")

app = FastAPI()

def get_team_data(team, year):
    response = supabase.table('march_madness_sq').select('*').eq('TEAM', team).eq('YEAR', year).execute()
    return pd.DataFrame(response.data)

@app.get("/simulate")
def simulate(team1: str, team2: str, team1_year: int, team2_year: int, num_simulations: int = 10000):
    df1 = get_team_data(team1, team1_year)
    df2 = get_team_data(team2, team2_year)

    if df1.empty or df2.empty:
        return {"error": "One or both teams not found"}



    row1, row2 = df1.iloc[0], df2.iloc[0]

    # Manually defined features (shortened example for brevity)
    features = pd.DataFrame({
    "diff_EFG_O": [row1["EFG_O"] - row2["EFG_O"]],
    "diff_EFG_D": [row1["EFG_D"] - row2["EFG_D"]],
    "diff_FTR": [row1["FTR"] - row2["FTR"]],
    "diff_FTRD": [row1["FTRD"] - row2["FTRD"]],
    "diff_ORB": [row1["ORB"] - row2["ORB"]],
    "diff_DRB": [row1["DRB"] - row2["DRB"]],
    "diff_TOR": [row1["TOR"] - row2["TOR"]],
    "diff_TORD": [row1["TORD"] - row2["TORD"]],
    "diff_X2P_O": [row1["X2P_O"] - row2["X2P_O"]],
    "diff_X3P_O": [row1["X3P_O"] - row2["X3P_O"]],
    "diff_X2P_D": [row1["X2P_D"] - row2["X2P_D"]],
    "diff_X3P_D": [row1["X3P_D"] - row2["X3P_D"]],
    "diff_dbpr_Player_1": [row1["dbpr_Player 1"] - row2["dbpr_Player 1"]],
    "diff_dbpr_Player_2": [row1["dbpr_Player 2"] - row2["dbpr_Player 2"]],
    "diff_dbpr_Player_3": [row1["dbpr_Player 3"] - row2["dbpr_Player 3"]],
    "diff_dbpr_Player_4": [row1["dbpr_Player 4"] - row2["dbpr_Player 4"]],
    "diff_dbpr_Player_5": [row1["dbpr_Player 5"] - row2["dbpr_Player 5"]],
    "diff_dbpr_Player_6": [row1["dbpr_Player 6"] - row2["dbpr_Player 6"]],
    "diff_dbpr_Player_7": [row1["dbpr_Player 7"] - row2["dbpr_Player 7"]],
    "diff_dbpr_Player_8": [row1["dbpr_Player 8"] - row2["dbpr_Player 8"]],
    "diff_obpr_Player_1": [row1["obpr_Player 1"] - row2["obpr_Player 1"]],
    "diff_obpr_Player_2": [row1["obpr_Player 2"] - row2["obpr_Player 2"]],
    "diff_obpr_Player_3": [row1["obpr_Player 3"] - row2["obpr_Player 3"]],
    "diff_obpr_Player_4": [row1["obpr_Player 4"] - row2["obpr_Player 4"]],
    "diff_obpr_Player_5": [row1["obpr_Player 5"] - row2["obpr_Player 5"]],
    "diff_obpr_Player_6": [row1["obpr_Player 6"] - row2["obpr_Player 6"]],
    "diff_obpr_Player_7": [row1["obpr_Player 7"] - row2["obpr_Player 7"]],
    "diff_obpr_Player_8": [row1["obpr_Player 8"] - row2["obpr_Player 8"]],
    "diff_Rim_and_3_rate": [row1["Rim_and_3_rate"] - row2["Rim_and_3_rate"]],
    "diff_X3PT_Frequency": [row1["X3PT.Frequency"] - row2["X3PT.Frequency"]],
    "diff_X3PT_SQ_PPP": [row1["X3PT.SQ.PPP"] - row2["X3PT.SQ.PPP"]],
    "diff_Catch_Shoot_3PT_Frequency": [row1["Catch...Shoot.3PT.Frequency"] - row2["Catch...Shoot.3PT.Frequency"]],
    "diff_Catch_Shoot_3PT_SQ_PPP": [row1["Catch...Shoot.3PT.SQ.PPP"] - row2["Catch...Shoot.3PT.SQ.PPP"]],
    "diff_Cut_Frequency": [row1["Cut.Frequency"] - row2["Cut.Frequency"]],
    "diff_Cut_SQ_PPP": [row1["Cut.SQ.PPP"] - row2["Cut.SQ.PPP"]],
    "diff_Finishing_at_the_Rim_Frequency": [row1["Finishing.at.the.Rim.Frequency"] - row2["Finishing.at.the.Rim.Frequency"]],
    "diff_Finishing_at_the_Rim_SQ_PPP": [row1["Finishing.at.the.Rim.SQ.PPP"] - row2["Finishing.at.the.Rim.SQ.PPP"]],
    "diff_Half_Court_Frequency": [row1["Half.Court.Frequency"] - row2["Half.Court.Frequency"]],
    "diff_Half_Court_SQ_PPP": [row1["Half.Court.SQ.PPP"] - row2["Half.Court.SQ.PPP"]],
    "diff_Isolation_Frequency": [row1["Isolation.Frequency"] - row2["Isolation.Frequency"]],
    "diff_Isolation_SQ_PPP": [row1["Isolation.SQ.PPP"] - row2["Isolation.SQ.PPP"]],
    "diff_Midrange_Frequency": [row1["Midrange.Frequency"] - row2["Midrange.Frequency"]],
    "diff_Midrange_SQ_PPP": [row1["Midrange.SQ.PPP"] - row2["Midrange.SQ.PPP"]],
    "diff_Off_the_Dribble_3PT_Frequency": [row1["Off.the.Dribble.3PT.Frequency"] - row2["Off.the.Dribble.3PT.Frequency"]],
    "diff_Off_the_Dribble_3PT_SQ_PPP": [row1["Off.the.Dribble.3PT.SQ.PPP"] - row2["Off.the.Dribble.3PT.SQ.PPP"]],
    "diff_Off_Screen_Frequency": [row1["Off.Screen.Frequency"] - row2["Off.Screen.Frequency"]],
    "diff_Off_Screen_SQ_PPP": [row1["Off.Screen.SQ.PPP"] - row2["Off.Screen.SQ.PPP"]],
    "diff_PR_Ball_Screen_Frequency": [row1["P.R.Ball.Screen.Frequency"] - row2["P.R.Ball.Screen.Frequency"]],
    "diff_PR_Ball_Screen_SQ_PPP": [row1["P.R.Ball.Screen.SQ.PPP"] - row2["P.R.Ball.Screen.SQ.PPP"]],
    "diff_Post_Up_Frequency": [row1["Post.Up.Frequency"] - row2["Post.Up.Frequency"]],
    "diff_Post_Up_SQ_PPP": [row1["Post.Up.SQ.PPP"] - row2["Post.Up.SQ.PPP"]],
    "diff_Transition_Frequency": [row1["Transition.Frequency"] - row2["Transition.Frequency"]],
    "diff_Transition_SQ_PPP": [row1["Transition.SQ.PPP"] - row2["Transition.SQ.PPP"]],
    "diff_SQ_PPP_Player_1": [row1["SQ.PPP_Player 1"] - row2["SQ.PPP_Player 1"]],
    "diff_SQ_PPP_Player_2": [row1["SQ.PPP_Player 2"] - row2["SQ.PPP_Player 2"]],
    "diff_SQ_PPP_Player_3": [row1["SQ.PPP_Player 3"] - row2["SQ.PPP_Player 3"]],
    "diff_SQ_PPP_Player_4": [row1["SQ.PPP_Player 4"] - row2["SQ.PPP_Player 4"]],
    "diff_SQ_PPP_Player_5": [row1["SQ.PPP_Player 5"] - row2["SQ.PPP_Player 5"]],
    "diff_SQ_PPP_Player_6": [row1["SQ.PPP_Player 6"] - row2["SQ.PPP_Player 6"]],
    "diff_SQ_PPP_Player_7": [row1["SQ.PPP_Player 7"] - row2["SQ.PPP_Player 7"]],
    "diff_SQ_PPP_Player_8": [row1["SQ.PPP_Player 8"] - row2["SQ.PPP_Player 8"]],
    "diff_Good_Possession_Rate_Player_1": [row1["Good_Possession_Rate_Player 1"] - row2["Good_Possession_Rate_Player 1"]],
    "diff_Good_Possession_Rate_Player_2": [row1["Good_Possession_Rate_Player 2"] - row2["Good_Possession_Rate_Player 2"]],
    "diff_Good_Possession_Rate_Player_3": [row1["Good_Possession_Rate_Player 3"] - row2["Good_Possession_Rate_Player 3"]],
    "diff_Good_Possession_Rate_Player_4": [row1["Good_Possession_Rate_Player 4"] - row2["Good_Possession_Rate_Player 4"]],
    "diff_Good_Possession_Rate_Player_5": [row1["Good_Possession_Rate_Player 5"] - row2["Good_Possession_Rate_Player 5"]],
    "diff_Good_Possession_Rate_Player_6": [row1["Good_Possession_Rate_Player 6"] - row2["Good_Possession_Rate_Player 6"]],
    "diff_Good_Possession_Rate_Player_7": [row1["Good_Possession_Rate_Player 7"] - row2["Good_Possession_Rate_Player 7"]],
    "diff_Good_Possession_Rate_Player_8": [row1["Good_Possession_Rate_Player 8"] - row2["Good_Possession_Rate_Player 8"]],
    "diff_Shot_Making_Player_1": [row1["Shot.Making_Player 1"] - row2["Shot.Making_Player 1"]],
    "diff_Shot_Making_Player_2": [row1["Shot.Making_Player 2"] - row2["Shot.Making_Player 2"]],
    "diff_Shot_Making_Player_3": [row1["Shot.Making_Player 3"] - row2["Shot.Making_Player 3"]],
    "diff_Shot_Making_Player_4": [row1["Shot.Making_Player 4"] - row2["Shot.Making_Player 4"]],
    "diff_Shot_Making_Player_5": [row1["Shot.Making_Player 5"] - row2["Shot.Making_Player 5"]],
    "diff_Shot_Making_Player_6": [row1["Shot.Making_Player 6"] - row2["Shot.Making_Player 6"]],
    "diff_Shot_Making_Player_7": [row1["Shot.Making_Player 7"] - row2["Shot.Making_Player 7"]],
    "diff_Shot_Making_Player_8": [row1["Shot.Making_Player 8"] - row2["Shot.Making_Player 8"]],
})
    
    # Predict win probability using XGBoost model
    dtest = xgb.DMatrix(features.values)
    prob = model.predict(dtest)[0]

    # Run simulations
    sims = np.random.rand(num_simulations)
    team1_wins = np.sum(prob >= sims)

    return {
        "team1": team1,
        "team1_year": team1_year,
        "team2": team2,
        "team2_year": team2_year,
        "team1_win_prob": round(team1_wins / num_simulations, 4),
        "team2_win_prob": round((num_simulations - team1_wins) / num_simulations, 4)
    }

@app.get("/preview")
def preview_data():
    try:
        response = supabase.table('march_madness_sq').select("*").limit(5).execute()
        print("Sample data:", response.data)  # View in terminal
        return response.data  # Will display in browser too
    except Exception as e:
        print("Error previewing data:", e)
        raise HTTPException(status_code=500, detail="Failed to preview data.")
