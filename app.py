import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Delivery Time Predictor (Location)", page_icon="📍", layout="wide")

DEFAULT_CSV = "food_delivery_data_with_location.csv"
TARGET = "Delivery_Time_min"
LOC_COLS = ["Restaurant_Lat", "Restaurant_Lon", "Customer_Lat", "Customer_Lon"]
AREA_COLS = ["Restaurant_Area", "Customer_Area"]
ID_COLS = ["Order_ID"]


# ---------------------------------------------------------------- helpers
def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance in km between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file).dropna().reset_index(drop=True)


@st.cache_resource
def train(df: pd.DataFrame, test_size: float):
    drop = [c for c in ID_COLS + LOC_COLS if c in df.columns]
    X = df.drop(columns=[TARGET] + drop)
    y = df[TARGET]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ]
    )
    model = Pipeline([("prep", pre), ("lr", LinearRegression())])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    metrics = {
        "MAE": mean_absolute_error(y_te, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
        "R2": r2_score(y_te, pred),
    }
    coefs = pd.Series(
        model.named_steps["lr"].coef_,
        index=model.named_steps["prep"].get_feature_names_out(),
    ).sort_values()
    results = pd.DataFrame({"Actual": y_te.values, "Predicted": pred})
    return model, metrics, coefs, results, X.columns.tolist()


# ---------------------------------------------------------------- sidebar
st.title("📍 Food Delivery Time Predictor — with Location")
st.caption("Pick pickup and drop-off locations; distance is calculated automatically from coordinates.")

with st.sidebar:
    st.header("Data & settings")
    upload = st.file_uploader("Upload CSV", type="csv")
    if upload:
        df = load_data(upload)
    elif os.path.exists(DEFAULT_CSV):
        df = load_data(DEFAULT_CSV)
        st.info(f"Using {DEFAULT_CSV}")
    else:
        st.error(f"Upload a CSV, or place {DEFAULT_CSV} next to this script.")
        st.stop()

    test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)
    road_factor = st.slider(
        "Road factor", 1.0, 2.0, 1.3, 0.05,
        help="Multiplies straight-line distance to approximate real road distance.",
    )

# ---------------------------------------------------------------- validate
required = [TARGET] + LOC_COLS + AREA_COLS
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"CSV is missing required columns: {', '.join(missing)}")
    st.stop()

if "Distance_km" not in df.columns:
    df["Distance_km"] = (
        haversine_km(df.Restaurant_Lat, df.Restaurant_Lon, df.Customer_Lat, df.Customer_Lon) * road_factor
    ).round(1)

model, metrics, coefs, results, feature_cols = train(df, test_size)

# Area centroids (average coordinates) used when the user picks an area name
pts = pd.concat(
    [
        df[["Restaurant_Area", "Restaurant_Lat", "Restaurant_Lon"]].set_axis(["Area", "lat", "lon"], axis=1),
        df[["Customer_Area", "Customer_Lat", "Customer_Lon"]].set_axis(["Area", "lat", "lon"], axis=1),
    ]
)
centroids = pts.groupby("Area")[["lat", "lon"]].mean()
areas = sorted(centroids.index)

# ---------------------------------------------------------------- tabs
tab_pred, tab_perf, tab_insights, tab_data = st.tabs(
    ["🔮 Predict", "📊 Model performance", "🗺 Location insights", "🗂 Data"]
)

with tab_pred:
    st.subheader("1 · Locations")
    c1, c2 = st.columns(2)
    r_area = c1.selectbox("Restaurant area", areas, index=0)
    c_area = c2.selectbox("Customer area", areas, index=min(1, len(areas) - 1))

    r_lat, r_lon = centroids.loc[r_area]
    c_lat, c_lon = centroids.loc[c_area]

    if st.checkbox("Enter exact coordinates"):
        a, b, c, d = st.columns(4)
        r_lat = a.number_input("Restaurant lat", value=float(r_lat), format="%.5f")
        r_lon = b.number_input("Restaurant lon", value=float(r_lon), format="%.5f")
        c_lat = c.number_input("Customer lat", value=float(c_lat), format="%.5f")
        c_lon = d.number_input("Customer lon", value=float(c_lon), format="%.5f")

    distance = round(float(haversine_km(r_lat, r_lon, c_lat, c_lon)) * road_factor, 1)
    distance = max(distance, 0.4)

    m1, m2 = st.columns([1, 2])
    m1.metric("Estimated road distance", f"{distance} km")
    m1.caption("Straight-line distance × road factor")
    m2.map(
        pd.DataFrame({"lat": [r_lat, c_lat], "lon": [r_lon, c_lon]}),
        size=120,
    )

    st.subheader("2 · Other conditions")
    other = [f for f in feature_cols if f not in AREA_COLS + ["Distance_km"]]
    cols = st.columns(3)
    inputs = {"Restaurant_Area": r_area, "Customer_Area": c_area, "Distance_km": distance}
    for i, f in enumerate(other):
        with cols[i % 3]:
            if pd.api.types.is_numeric_dtype(df[f]):
                is_int = pd.api.types.is_integer_dtype(df[f])
                lo, hi = df[f].min(), df[f].max()
                inputs[f] = st.number_input(
                    f.replace("_", " "),
                    min_value=int(lo) if is_int else float(lo),
                    max_value=int(hi) if is_int else float(hi),
                    value=int(df[f].median()) if is_int else float(df[f].median()),
                )
            else:
                inputs[f] = st.selectbox(f.replace("_", " "), sorted(df[f].unique()))

    if st.button("Predict delivery time", type="primary"):
        est = float(model.predict(pd.DataFrame([inputs])[feature_cols])[0])
        mae = metrics["MAE"]
        k1, k2 = st.columns(2)
        k1.metric("Estimated delivery time", f"{est:.0f} min")
        k2.metric("Likely range", f"{max(est - mae, 0):.0f} – {est + mae:.0f} min", help="Prediction ± test-set MAE")

with tab_perf:
    k1, k2, k3 = st.columns(3)
    k1.metric("MAE", f"{metrics['MAE']:.2f} min")
    k2.metric("RMSE", f"{metrics['RMSE']:.2f} min")
    k3.metric("R²", f"{metrics['R2']:.3f}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Actual vs predicted (test set)**")
        st.scatter_chart(results, x="Actual", y="Predicted")
    with right:
        st.markdown("**Feature coefficients**")
        st.bar_chart(coefs)

with tab_insights:
    left, right = st.columns(2)
    with left:
        st.markdown("**Average delivery time by customer area (min)**")
        st.bar_chart(df.groupby("Customer_Area")[TARGET].mean().sort_values())
    with right:
        st.markdown("**Average delivery time by restaurant area (min)**")
        st.bar_chart(df.groupby("Restaurant_Area")[TARGET].mean().sort_values())
    st.markdown("**Delivery locations (customers)**")
    st.map(df.rename(columns={"Customer_Lat": "lat", "Customer_Lon": "lon"})[["lat", "lon"]])

with tab_data:
    st.write(f"{len(df):,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(200))
