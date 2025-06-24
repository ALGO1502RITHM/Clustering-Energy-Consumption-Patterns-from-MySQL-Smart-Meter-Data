import pandas as pd
import numpy as np

# ---load datasets---
Energy_Consumption_data = pd.read_csv('household_power_consumption.txt', sep= ';', parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True,
                                      na_values='?', low_memory=False)

# ---Energy Consumption Info---
# print(Energy_Consumption_data.info())
# print(Energy_Consumption_data.isnull().sum())
# print(Energy_Consumption_data.describe())
# print(Energy_Consumption_data['Sub_metering_1'])

# ---Dropping rows with missing values---
Energy_Consumption_data.dropna(inplace=True)
# print(Energy_Consumption_data.head())

# ---Resampling to hourly avarage---
Energy_Consumption_data.set_index('datetime', inplace=True)
Energy_Consumption_data_hourly = Energy_Consumption_data.resample('H').mean().dropna().reset_index()
# print(Energy_Consumption_data_hourly['datetime'].head())
# print(Energy_Consumption_data_hourly.head())

# ---Featuring Engineering---
# ---hour feature---
Energy_Consumption_data_hourly['hour'] = Energy_Consumption_data_hourly['datetime'].dt.hour

# ---is_weekend feature---
Energy_Consumption_data_hourly['day_of_week'] = Energy_Consumption_data_hourly['datetime'].dt.dayofweek
Energy_Consumption_data_hourly['is_weekend'] = Energy_Consumption_data_hourly['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# ---total_Submetering feature---
Energy_Consumption_data_hourly['total_submetering'] = (Energy_Consumption_data_hourly['Sub_metering_1'] + Energy_Consumption_data_hourly['Sub_metering_2'] + Energy_Consumption_data_hourly['Sub_metering_3'])

# ---time of day feature---
bins = [0, 6, 12, 18, 24]
labels =['Night', 'Morning', 'Afternoon', 'Evening']
Energy_Consumption_data_hourly['time_of_day'] = pd.cut(Energy_Consumption_data_hourly['hour'], bins=bins, labels=labels, right=False)
# print(Energy_Consumption_data_hourly['time_of_day'].head())

# ---Submetering_ratio feature---
Energy_Consumption_data_hourly['Submetering_ratio'] = (Energy_Consumption_data_hourly['total_submetering'] / (Energy_Consumption_data_hourly['Global_active_power'] * 1000 / 60))

# ---Appliance ratio---
Energy_Consumption_data_hourly['Kitchen_ratio'] = Energy_Consumption_data_hourly['Sub_metering_2'] / Energy_Consumption_data_hourly['total_submetering']
Energy_Consumption_data_hourly['ac_ratio'] = Energy_Consumption_data_hourly['Sub_metering_3'] / Energy_Consumption_data_hourly['total_submetering']
# print(Energy_Consumption_data_hourly['Kitchen_ratio'].head())
# print(Energy_Consumption_data_hourly['ac_ratio'].head())

print(Energy_Consumption_data_hourly.head())

# ---save to csv file---
Energy_Consumption_data_hourly.to_csv("energy_consumption_preprocessing.csv", index=False)
