import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Real Estate Price Estimator", page_icon="🏠", layout="wide")

if "realestate_history" not in st.session_state:
    st.session_state.realestate_history = []

st.title("🏠 Real Estate Price Estimator")

# ─── DATASET (NO NULL VALUES) ───
np.random.seed(42)
n = 50

data = {
    "PropertyID": [f"P{str(i).zfill(3)}" for i in range(1, n+1)],
    "City": (["Hyderabad","Bangalore","Mumbai","Pune","Chennai"] * (n//5)),
    "Locality": (["Prime","Mid","Suburban","Outskirts"] * (n//4 + 1))[:n],

    # ❌ Removed None
    "AreaSqFt": np.random.choice([800,1200,1500,2000,2500,3000], n),
    "Bedrooms": np.random.choice([1,2,3,4,5], n),
    "Bathrooms": np.random.choice([1,2,3,4], n),
    "AgeYears": np.random.choice([1,5,10,15,20,25], n),
    "ParkingSpots": np.random.choice([0,1,2,3], n),

    # ❌ Removed None
    "NearMetroKm": np.random.choice([0.5,1,2,3,5], n),

    "FurnishingStatus": (["Furnished","Semi","Unfurnished"] * (n//3 + 1))[:n],
    "FloorNumber": np.random.randint(0, 20, n),
}

df_raw = pd.DataFrame(data)

# Target generation
df_raw["PriceLakh"] = (
    df_raw["AreaSqFt"] * 0.08 +
    df_raw["Bedrooms"] * 10 +
    df_raw["Bathrooms"] * 5 -
    df_raw["AgeYears"] * 1.5 +
    np.random.randint(-20, 20, n)
)

st.subheader("📋 Dataset")
st.dataframe(df_raw)

# ─── PREPROCESSING ───
df = df_raw.copy()

# ❌ Removed fillna (not needed now)
df.drop_duplicates(inplace=True)
df.drop("PropertyID", axis=1, inplace=True)

# Encoding
df["City"] = df["City"].map({"Hyderabad":0,"Bangalore":1,"Mumbai":2,"Pune":3,"Chennai":4})
df["Locality"] = df["Locality"].map({"Prime":3,"Mid":2,"Suburban":1,"Outskirts":0})
df["FurnishingStatus"] = df["FurnishingStatus"].map({"Furnished":2,"Semi":1,"Unfurnished":0})

# ─── FEATURE ENGINEERING ───
df["AgeDepreciation"] = 1 / (df["AgeYears"] + 1)
df["ConnectivityScore"] = 1 / (df["NearMetroKm"] + 0.1)
df["RoomDensity"] = (df["Bedrooms"] + df["Bathrooms"]) / (df["AreaSqFt"] / 100)

# ─── MODEL ───
X = df.drop("PriceLakh", axis=1)
y = df["PriceLakh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Lasso(alpha=0.5, max_iter=5000)
model.fit(X_train, y_train)

st.success("✅ Model Trained!")

# ─── EVALUATION ───
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Evaluation")
st.write({
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
})

# Plot
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

# ─── PREDICTION ───
st.subheader("🎯 Predict Price")

city = st.selectbox("City", [0,1,2,3,4])
locality = st.selectbox("Locality", [0,1,2,3])
area = st.slider("Area", 500, 5000, 1500)
beds = st.slider("Bedrooms", 1, 6, 3)
baths = st.slider("Bathrooms", 1, 4, 2)
age = st.slider("Age", 0, 30, 5)
parking = st.slider("Parking", 0, 3, 1)
metro = st.slider("Metro Distance", 0.1, 10.0, 2.0)
furnish = st.selectbox("Furnishing", [0,1,2])
floor = st.slider("Floor", 0, 20, 5)

if st.button("Predict"):
    age_dep = 1 / (age + 1)
    conn = 1 / (metro + 0.1)
    room = (beds + baths) / (area / 100)

    input_df = pd.DataFrame([[city, locality, area, beds, baths, age,
                               parking, metro, furnish, floor,
                               age_dep, conn, room]],
                            columns=X.columns)

    pred = model.predict(input_df)[0]

    st.success(f"🏠 Estimated Price: ₹{pred:.2f} Lakh")