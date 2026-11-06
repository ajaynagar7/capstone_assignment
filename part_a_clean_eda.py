import pandas as pd
import numpy as np
import src.shipment_prep as sp

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

raw_shipments = pd.read_csv("data/raw_shipments.csv")
raw_shipments_without_duplicate = raw_shipments.drop_duplicates()
hubs = pd.read_csv("data/hubs.csv")

clean, median = sp.clean_shipments(raw_shipments)
features = sp.build_features(clean, hubs)
features["is_delayed"] = np.where(features["delivery_hours"] > features["promised_hours"], 1, 0)

modelling_data = features[FEATURES]
modelling_data["delivery_hours"] = features["delivery_hours"]
modelling_data["is_delayed"] = features["is_delayed"]
modelling_data.to_csv("data/modelling_table.csv", index=False)

# # A5. Checks file.


hours_q1 = features["delivery_hours"].quantile(0.25)
hours_q3 = features["delivery_hours"].quantile(0.75)
hours_lower_bound = hours_q1 - 1.5 * (hours_q3 - hours_q1)
hours_upper_bound = hours_q3 + 1.5 * (hours_q3 - hours_q1)
metrices = {"raw_rows": len(raw_shipments),
      "duplicate_rows_removed" : raw_shipments.duplicated().sum(),
      "rows_after_dedup" : len(raw_shipments_without_duplicate),
      "missing_distance_km_before" : raw_shipments_without_duplicate["distance_km"].isna().sum(),
      "missing_parcel_weight_before" : raw_shipments_without_duplicate["parcel_weight_kg"].isna().sum(),
      "sentinel_weight_rows" : len(raw_shipments_without_duplicate[raw_shipments_without_duplicate["parcel_weight_kg"] >= 40]),
      "missing_declared_value_before" : raw_shipments_without_duplicate["declared_value"].isna().sum(),
      "missing_payment_mode_before" : raw_shipments_without_duplicate["payment_mode"].isna().sum(),
      "service_type_spellings_before" : len(raw_shipments_without_duplicate["service_type"].unique()),
      "service_type_spellings_after" : len(features["service_type"].unique()),
      "declared_value_mean_inr" : round(features["declared_value_inr"].mean(), 4),
      "log_declared_value_mean" : round(features["log_declared_value"].mean(), 4),
      "median_distance_km" : median["distance_km"],
      "median_parcel_weight_kg" : median["parcel_weight_kg"],
      "median_declared_value_inr" : median["declared_value_inr"],
      "rows_after_hub_merge" : len(features),
      "nulls_in_features_and_target" : features.isnull().sum().sum() + features["delivery_hours"].isnull().sum().sum(),
      "delayed_count" : np.sum(features["is_delayed"] == 1),
      "delay_rate" : round(np.sum(features["is_delayed"] == 1) / len(features), 4),
      "hours_q1 / hours_q3" : f"{round(hours_q1, 4)} , / {round(hours_q3, 4)}",
      "hours_lower_bound / hours_upper_bound" : f"{round(hours_lower_bound, 4)} / {round(hours_upper_bound, 4)}",
      "hours_outlier_count" : np.sum((features["delivery_hours"] < hours_lower_bound) + (features["delivery_hours"] > hours_upper_bound)),
      "declared_value_median_inr" : median["declared_value_inr"],
      "log_declared_value_median" : round(features["log_declared_value"].median(), 4)


      }

part_a_checks = pd.DataFrame({"metric" : names, "value" : values} for names, values in metrices.items())

part_a_checks.to_csv("outputs/part_a_checks.csv")

# # 6 Delay evidence table

desired_order = ["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]

grouped = round(features.groupby("service_type")["is_delayed"].agg(shipments = "count", delayed = "sum", delay_rate = "mean"),4).reset_index()
groupedp = round(features.groupby("service_type")["delivery_hours"].agg(mean_hours = "mean"),4).reset_index()
grouped = pd.merge(grouped, groupedp, on="service_type")
grouped["dimension"] = "service_type"
grouped["group"] = grouped["service_type"]
grouped = grouped.sort_values(by="mean_hours", ascending=True)
grouped = grouped[desired_order]


grouped1  = round(features.groupby("hub_tier")["is_delayed"].agg(shipments = "count", delayed = "sum", delay_rate = "mean"),4).reset_index()
grouped1["dimension"] = "hub_tier"
grouped1["group"] = grouped1["hub_tier"]
groupedp1 = round(features.groupby("hub_tier")["delivery_hours"].agg(mean_hours = "mean"), 4).reset_index()
grouped1 = pd.merge(grouped1, groupedp1, on="hub_tier")
grouped1 = grouped1[desired_order]


bins = [0, 200, 400, 600, 2000]
features["dist_bins"] = pd.cut(features["distance_km"], bins=bins)
grouped3 = round(features.groupby("dist_bins")["is_delayed"].agg(shipments = "count", delayed = "sum", delay_rate = "mean"),4).reset_index()
grouped3["dimension"] = "distance_band"
grouped3["group"] = grouped3["dist_bins"]
groupedp3 = round(features.groupby("dist_bins")["delivery_hours"].agg(mean_hours = "mean"), 4).reset_index()
grouped3 = pd.merge(grouped3, groupedp3, on="dist_bins")
grouped3["group"] = grouped3["group"].astype(str).str.replace("(","", regex=False).str.replace("]", "", regex=False).str.replace(",","-", regex=False).str.replace("- 2000", "+", regex=False)
grouped3 = grouped3[desired_order]


grouping = pd.concat([grouped, grouped1, grouped3], ignore_index=True)


grouping.to_csv("outputs/part_a_delay_by_group.csv", index=False)

correlation = features.select_dtypes(include=["number"])
correlation_matrix = round(correlation.corr(),4)
correlation_matrix.to_csv("outputs/part_a_correlations.csv", index=False)

r = correlation_matrix.loc["delivery_hours", "promised_hours"]
s = correlation_matrix.loc["delivery_hours", "distance_km"]
t = correlation_matrix.loc["is_delayed", "distance_km"]
print(f"delivery_hours and promised_hours correlation is :{r}\ndelivery_hours and distance_km correlation is :{s}\nis_delayed and distance_km correlation is : {t}")

# mean_before = features["declared_value_inr"].mean()
# median_before = features["declared_value_inr"].median()
# mean_after = features["log_declared_value"].mean()
# median_after = features["log_declared_value"].median()
# print(mean_before)
# print(median_before)
# print(mean_after)
# print(median_after)