import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import json

df = pd.read_csv('bengaluru_house_prices (1).csv')
df = df.drop(['area_type', 'availability', 'society', 'balcony'], axis=1)
df = df.dropna()
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
df = df.drop('size', axis=1)

def convert_sqft(x):
    try:
        if '-' in str(x):
            vals = x.split('-')
            return (float(vals[0]) + float(vals[1])) / 2
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna()
df = df[df['total_sqft'] / df['bhk'] >= 300]

loc_counts = df['location'].value_counts()
top_locations = loc_counts[loc_counts > 10].index
df['location'] = df['location'].apply(lambda x: x if x in top_locations else 'other')

# One-hot encode manually
dummies = pd.get_dummies(df['location'])
X = pd.concat([dummies, df[['total_sqft', 'bath', 'bhk']]], axis=1)
y = df['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Ridge()
model.fit(X_scaled, y)

locations = sorted(df['location'].unique().tolist())
columns = list(dummies.columns) + ['total_sqft', 'bath', 'bhk']

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
with open('locations.json', 'w') as f:
    json.dump({'locations': locations, 'columns': columns}, f)

print("✅ Done!")