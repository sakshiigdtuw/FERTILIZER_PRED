import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset
df = pd.read_csv("C:/Users/sukri/OneDrive/Desktop/fertilizer/Fertilizer Prediction.csv")

# Encode categorical features
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])
df['Fertilizer Name'] = le_fert.fit_transform(df['Fertilizer Name'])

X = df.drop('Fertilizer Name', axis=1)
y = df['Fertilizer Name']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model and encoders
pickle.dump(model, open('fertilizer_model.pkl', 'wb'))
pickle.dump(le_soil, open('soil_encoder.pkl', 'wb'))
pickle.dump(le_crop, open('crop_encoder.pkl', 'wb'))
pickle.dump(le_fert, open('fert_encoder.pkl', 'wb'))

print("Model and encoders saved successfully.")
