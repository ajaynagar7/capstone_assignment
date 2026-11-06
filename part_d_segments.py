import pandas as pd
import numpy as np
import src.shipment_prep as sp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# 1 Hub profile

raw_shipments = pd.read_csv("data/raw_shipments.csv")
raw_shipments_without_duplicate = raw_shipments.drop_duplicates()
hubs = pd.read_csv("data/hubs.csv")

clean, median = sp.clean_shipments(raw_shipments)
features = sp.build_features(clean, hubs)
features["is_delayed"] = np.where(features["delivery_hours"] > features["promised_hours"], 1, 0)

FEATURES_hubs = ["hub_id", "delivery_hours", "is_delayed", "distance_km", "n_stops"]

profile = features.groupby("hub_id").agg({"promised_hours" : "count",
                                    "delivery_hours" : "mean", 
                                    "is_delayed" : "mean", 
                                    "distance_km" : "mean",
                                    "n_stops" : "mean"})


profile = profile.rename(columns={"promised_hours" : "shipments",
                                  "distance_km" : "mean_distance_km", 
                                  "delivery_hours" : "mean_delivery_hours", 
                                  "is_delayed" : "delay_rate", 
                                  "n_stops" : "mean_n_stops"
                                  })


profile.to_csv("outputs/hub_profile.csv")

# 2 K-Means segmentation.

columns = profile[["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]]
scaler = StandardScaler()
scaled_columns = scaler.fit_transform(columns)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster = model.fit_predict(scaled_columns)
profile["lables"] = model.labels_





def mean_within_distance(k, scaled_data):
    
    mean_distance_from_clusters_within = []
    cluster_size = []
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    lables = model.fit_predict(scaled_data)
    cluster_centers = model.cluster_centers_
    for i in range(k):
        points_in_cluster = scaled_data[lables==i]
        dist_from_center = np.linalg.norm(points_in_cluster - cluster_centers[i]).mean()
        mean_distance_from_clusters_within.append(dist_from_center)
        cluster_size.append(len(points_in_cluster))

   

    
    dist_matrix = pairwise_distances(cluster_centers, metric="euclidean")
    across_distances = dist_matrix[np.triu_indices(k, k=1)]

    across_over_within = np.sum(across_distances)/np.sum(mean_distance_from_clusters_within)
    cluster_size.sort(reverse=True)
    return {
        "K" : k,
        "across_over_within" : across_over_within,
        "cluster_size" : cluster_size
        }

d_separation = []
for k in [2,3,4,5]:
    d_separation.append(mean_within_distance(k, scaled_columns))

print(pd.DataFrame(d_separation))

# for k in [2,3,4,5]:
#     model = KMeans(n_clusters=k, random_state=42, n_init=10)
#     model.fit(scaled_columns)
#     lables = model.labels_
#     centers = model.cluster_centers_
#     points_in_cluster = scaled_columns[lables == ]





# dist_matrix = pairwise_distances(scaled_profile, metric="euclidean")

# hubs = scaled_profile.shape[0]
# triu_i, triu_j = np.triu_indices(hubs, k=1)
# pair_dist = pairwise_distances([triu_i, triu_j])

# results = []
# for k in [2,3,4,5]:
#     km = KMeans(n_clusters=k, random_state=42, n_init=10)
#     lables = km.fit_predict(scaled_profile)

#     same_cluster_mask = lables[triu_i] == [triu_j]
#     diff_cluster_mask = ~same_cluster_mask

#     within_dist = pairwise_distances([same_cluster_mask])
#     across_dist = pairwise_distances([diff_cluster_mask])

#     mean_within = within_dist.mean()
#     mean_across = across_dist.mean()
#     ratio = mean_across/mean_within

#     results.append({
#         "K": k,
#         "n_within_pairs": int(same_cluster_mask.sum()),
#         "n_across_pairs": int(diff_cluster_mask.sum()),
#         "mean_within_dist": round(float(mean_within), 4),
#         "mean_across_dist": round(float(mean_across), 4),
#         "across_to_within_ratio": round(float(ratio), 4)
#     })

# print(pd.DataFrame(results))
