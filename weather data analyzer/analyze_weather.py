import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('weather_data.csv')
df.columns = df.columns.str.strip() 

df ['Date_Time'] = pd.to_datetime(df [ 'Date_Time']) 

df.sort_values('Date_Time',inplace=True) 

sns.set(style="whitegrid")

plt.figure(figsize=(10, 5))
plt.plot(df['Date_Time'],df['Temperature_C'], marker='o',linestyle='-', color='blue')
plt.title('Temperature Trend Over Time (2024)')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,5))
plt.bar(df['Date_Time'],df['Precipitation_mm'], color='skyblue')
plt.title('Monthly Rainfall Trend (2024)')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='Humidity_pct',y='Temperature_C', data=df)
plt.title('Humidity vs Temperature (2024)')
plt.xlabel('Humidity (%)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.tight_layout()
plt.show()

df ['Date_Ordinal'] = df['Date_Time'].map(pd.Timestamp.toordinal)
X = df [['Date_Ordinal']]
y = df ['Temperature_C']
model = LinearRegression()
model.fit(X, y)
df ['Predicted_Temp'] = model.predict(X)

plt.figure(figsize=(10,5))
plt.plot(df ['Date_Time'],df['Temperature_C'], label='Actual Temperature', color='blue')
plt.plot(df['Date_Time'],df['Predicted_Temp'], label='PredictedTrend', color='orange')
plt.title('Temperature Forecast Using Linear Regression')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nStatistical Summary for 2024 Climate Data:")
print(df[['Temperature_C','Humidity_pct','Precipitation_mm']].describe())