# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ✅ Enable local cache
fastf1.Cache.enable_cache(r"D:\PYTON PROGRAMMING\PYTHON FILES\Data-Visualization-Using-Python\STREAMLIT & PANEL\F1 RACE ANALYSIS PROJECTS\F1 Candian GP 2025\cache")

# 🏁 Define season and sessions
year = 2025
session_types = ['FP1', 'FP2', 'FP3']
target_driver = 'VER'  # Use 3-letter driver code

# 🧱 Initialize storage
all_laps = []
telemetry_by_lap = []

# 📦 Loop through each Free Practice session
for session_type in session_types:
    try:
        session = fastf1.get_session(year, 'Canadian GP', session_type)
        session.load()

        driver_laps = session.laps.pick_driver(target_driver).copy()
        driver_laps['SessionType'] = session_type
        all_laps.append(driver_laps)

        # 🔁 Loop through each lap and get telemetry
        for _, lap in driver_laps.iterrows():
            try:
                # Get the actual FastF1 Lap object
                lap_obj = session.laps[
                    (session.laps['Driver'] == lap['Driver']) &
                    (session.laps['LapNumber'] == lap['LapNumber'])
                ].iloc[0]

                # Only proceed if LapTime is valid
                if pd.notna(lap_obj['LapTime']):
                    tel = lap_obj.get_telemetry()
                    tel['LapNumber'] = lap['LapNumber']
                    tel['SessionType'] = session_type
                    telemetry_by_lap.append(tel)
            except Exception as e:
                print(f"⚠️ Skipping lap {lap['LapNumber']} in {session_type}: {e}")
    except Exception as e:
        print(f"❌ Error loading session {session_type}: {e}")

# 🧩 Concatenate lap and telemetry DataFrames
laps_df = pd.concat(all_laps, ignore_index=True)
tel_df = pd.concat(telemetry_by_lap, ignore_index=True)

# %%
weather = session.weather_data
weather = weather.drop(79)
laps_df[['TrackTemp','Rainfall']] = weather[['TrackTemp', 'Rainfall']]
laps_df['LapTimeSeconds'] = laps_df['LapTime'].dt.total_seconds()

# Filter the data frame by the required columns 
tel_filtered = tel_df[['LapNumber', 'SessionType', 'Brake', 'Throttle', 'nGear', 'DRS']]
processed_df = laps_df[['LapNumber', 'SessionType','Compound', 'TyreLife', 'LapTimeSeconds', 'TrackTemp','Rainfall']]

# %%
# Data Processing  
for session in session_types:
    for lap in tel_filtered[tel_filtered['SessionType'] == session]['LapNumber'].unique():
        filter_data = tel_filtered[
            (tel_filtered['SessionType'] == session) & (tel_filtered['LapNumber'] == lap)
            ]
        # Throttle
        mean = np.mean(filter_data['Throttle'])
        std = np.std(filter_data['Throttle'])
        
        processed_df.loc[(processed_df['SessionType'] == session) & (processed_df['LapNumber'] == lap), 'mean_throttle'] = mean
        processed_df.loc[(processed_df['SessionType'] == session) & (processed_df['LapNumber'] == lap), 'std_throttle'] = std
        
        # Gear
        mean_gear = np.mean(filter_data['nGear'])
        std_gear = np.mean(filter_data['nGear'])
        gear_arr = np.array(filter_data['nGear'])
        values, counts = np.unique(gear_arr, return_counts=True)
        gear_mode = values[np.argmax(counts)]
        max_gear = np.max(filter_data['nGear'])
        
        processed_df.loc[(processed_df['SessionType'] == session) & (processed_df['LapNumber'] == lap), 'mean_gear'] = mean_gear
        processed_df.loc[(processed_df['SessionType'] == session) & (processed_df['LapNumber'] == lap), 'std_gear'] = std_gear
        processed_df.loc[(processed_df['SessionType'] == session) & (processed_df['LapNumber'] == lap), 'mode_gear'] = gear_mode
        processed_df.loc[(processed_df['SessionType'] == session) & (processed_df['LapNumber'] == lap), 'max_gear'] = max_gear
        
        # DRS
        drs_flag = filter_data['DRS'].apply(lambda x: 1 if x > 10 else 0).sum()
        processed_df.loc[(processed_df['SessionType'] == session) & (processed_df['LapNumber'] == lap), 'DRS'] = drs_flag
         
        # Brake 
        mean_brake = (filter_data['Brake'].sum()) / len(filter_data['Brake'])
        processed_df.loc[(processed_df['SessionType'] == session) & (processed_df['LapNumber'] == lap), 'mean_brake'] = mean_brake
        
processed_df['Rainfall'] = processed_df['Rainfall'].map({True : 1 , False : 0})
processed_df['Compound'] = processed_df['Compound'].map({'SOFT' : 6, 'MEDIUM' : 5, 'HARD' : 4})
processed_df.dropna(inplace=True)

# %%
processed_df.tail()

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, f1_score

# Model Training
X = processed_df[['Compound', 'TyreLife', 'TrackTemp', 'Rainfall', 'mean_throttle', 'std_throttle', 'mean_gear', 'std_gear', 'mode_gear', 'max_gear', 'DRS', 'mean_brake']].to_numpy()
Y = processed_df['LapTimeSeconds']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.3, random_state = 42) 

model = LinearRegression()
model.fit(train_X, train_Y)
predict = model.predict(test_X)
linear_r2 = r2_score(test_Y, predict)
print(linear_r2)

# %%

model = LinearRegression()
model.fit(X, Y)

r2_score = model.score(X, Y)
print("R² score:", r2_score)

predict = model.predict([[4, 1, 45, 0, 74, 40, 5, 5, 4, 8, 156, 0.15]])
print(predict)


