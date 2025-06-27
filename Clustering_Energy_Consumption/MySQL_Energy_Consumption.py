import mysql.connector
import pandas as pd

# ---Load Preprocessed data---
df_preprocess_data = pd.read_csv('energy_consumption_preprocessing.csv', parse_dates=['datetime'])

# ---Connect to MySQL---
connection = mysql.connector.connect(
    host="localhost",
    user="energy_user_",
    password="algo1502rithm",
    database="energy_consumption_database"
)
cursor = connection.cursor()

# ---Create Table in Database---
cursor.execute("""
    CREATE TABLE IF NOT EXISTS energy_consumption (
        id INT AUTO_INCREMENT PRIMARY KEY, 
        datetime DATETIME,
        Global_active_power FLOAT,
        hour INT,
        is_weekend INT,
        total_submetering FLOAT,
        time_of_day VARCHAR(20),
        submetering_ratio FLOAT,
        Kitchen_ratio FLOAT,
        ac_ratio FLOAT       
    );
""")

# ---Inserting rows one by one---
for _, row in df_preprocess_data.iterrows():
    cursor.execute("""
        INSERT INTO energy_consumption (
            datetime, Global_active_power, hour, is_weekend,
            total_submetering, time_of_day, Submetering_ratio,
            Kitchen_ratio, ac_ratio
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
""", (
    row['datetime'].to_pydatetime(), # ---timestamp convert---
    row['Global_active_power'],
    row['hour'],
    row['is_weekend'],
    row['total_submetering'],
    row['time_of_day'],
    row['Submetering_ratio'],
    row['Kitchen_ratio'],
    row['ac_ratio']
))

# ---commit and close---
connection.commit()
cursor.close()
connection.close()