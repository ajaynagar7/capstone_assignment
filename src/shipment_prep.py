import pandas as pd
import numpy as np

raw_data = pd.read_csv("data/raw_shipments.csv")


def clean_shipments(raw, medians=None):
    df = raw.copy()
    df = df.drop_duplicates()
    df["service_type"] = df["service_type"].str.strip().str.title()
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    df["declared_value_inr"] = df["declared_value"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False)
    df["declared_value_inr"] = pd.to_numeric(df["declared_value_inr"], errors="coerce")
    df = df.drop(columns=["declared_value"])
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], format="%Y-%m-%d %H:%M", errors="coerce")
    df["parcel_weight_kg"] = np.where(df["parcel_weight_kg"] > 100, np.nan, df["parcel_weight_kg"])

    medians = {
        "declared_value_inr": round(df["declared_value_inr"].median(), 4),
        "parcel_weight_kg": round(df["parcel_weight_kg"].median(), 4),
        "distance_km": round(df["distance_km"].median(), 4)
    }
    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    df["distance_km"] = df["distance_km"].fillna(medians["distance_km"])
    df["declared_value_inr"] = df["declared_value_inr"].fillna(medians["declared_value_inr"])
    return df, medians

# print(raw_data.columns.to_list())

hubs = pd.read_csv("data/hubs.csv")
print(hubs.info())


PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def build_features(clean, hubs):
    df = pd.merge(clean, hubs, how="left", on="hub_id")  
    
    df["promised_hours"] = clean["service_type"].map(PROMISED_HOURS)
    df["is_cod"] = np.where(clean["payment_mode"].str.lower() == "cod", 1, 0)
    df["is_weekend"] = np.where(clean["dispatch_time"].dt.dayofweek >= 5, 1, 0)
    df["dispatch_hour"] = clean["dispatch_time"].dt.hour
    df["log_declared_value"] = np.log1p(clean["declared_value_inr"])

    clean_df = df
    
       
    return clean_df

