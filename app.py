from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load("xgboost_multioutput.pkl")

# Expected feature list
expected_features = [
    "District", "Crop", "Year", "Area Sown (ha)", 
    "Production (MT)", "Yield (MT/ha)", "Rainfall (mm)", "Temperature (°C)", 
    "Humidity (%)"
]

# Mapping district and crop names to IDs
district_mapping = {"Pune": 1, "Nashik": 2, "Aurangabad": 3, "Nagpur": 4, "Satara": 5}
crop_mapping = {"Wheat": 1, "Rice": 2, "Maize": 3, "Soybean": 4, "Sugarcane": 5,
                "Barley": 6, "Cotton": 7, "Pulses": 8}

@app.post("/predict")
def predict_price(user_input: dict):
    """Convert user-input names to IDs and predict prices."""
    try:
        # Convert District and Crop Names to IDs
        if user_input["District"] in district_mapping:
            user_input["District"] = district_mapping[user_input["District"]]
        else:
            raise HTTPException(status_code=400, detail="Invalid District Name")

        if user_input["Crop"] in crop_mapping:
            user_input["Crop"] = crop_mapping[user_input["Crop"]]
        else:
            raise HTTPException(status_code=400, detail="Invalid Crop Name")

        # Convert input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Predict prices
        prediction = model.predict(input_df)

        return {
            "Farm Gate Price (INR/quintal)": float(prediction[0][0]),
            "Wholesale Price (INR/quintal)": float(prediction[0][1]),
            "Retail Price (INR/quintal)": float(prediction[0][2])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
