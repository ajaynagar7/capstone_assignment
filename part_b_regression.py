import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


df = pd.read_csv("data\modelling_table.csv")

X = df.drop(columns=["is_delayed", "delivery_hours"])
y = df["delivery_hours"]

# 1. Split into training, validation and test.

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. Scaling without leakage.
feture_scaler = StandardScaler()
target_scaler = StandardScaler()
X_train_scaled = feture_scaler.fit_transform(X_train)
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
X_val_scaled = feture_scaler.transform(X_val)
y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1))
X_test_scaled = feture_scaler.transform(X_test)
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

metrics = [
    ("train_rows", len(X_train_scaled)),
    ("val_rows" , len(X_val_scaled)),
    ("test_rows" , len(X_test_scaled)),
    ("train_scaled_max_abs_mean" , np.mean(np.abs(X_train_scaled))),
    ("test_scaled_max_abs_mean" , np.mean(np.abs(X_test_scaled)))
]

b_check = pd.DataFrame([{"metric" : metric, "value" : value} for metric, value in metrics])

b_check.to_csv("outputs/part_b_checks.csv", index=False)

# 3 Baseline.
base_train_pred = y_train.mean()
base_val_pred = np.full(shape=len(y_val), fill_value=base_train_pred)
base_val_mae = mean_absolute_error(y_val, base_val_pred)
base_val_mse = mean_squared_error(y_val, base_val_pred)
base_val_rmse = root_mean_squared_error(y_val, base_val_pred)
base_val_r2 = r2_score(y_val, base_val_pred)

print(f"base_val_mae : {round(base_val_mae,4)}\nbase_val_mse : {round(base_val_mse,4)}\n base_val_rmse : {round(base_val_rmse,4)}")

# 4 Tree-depth diagnostic.
tree_depth = []
depths = [1, 2, 3, 4, 6, 8, 12, None]
for d in depths:
    model = DecisionTreeRegressor(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    y_val_pred = model.predict(X_val)
    val_rmse = root_mean_squared_error(y_val, y_val_pred)
    tree_depth.append({
        "max_depth" : "Unlimited" if d is None else d,
        "train_rmse" : round(train_rmse,4),
        "val_rmse" : round(val_rmse,4)
    })

b_tree_depth = pd.DataFrame(tree_depth)
b_tree_depth.to_csv("outputs/part_b_tree_depth.csv", index=False)

# 5 Four candidate models


models = [("linear_regression", LinearRegression()), ("decision_tree", DecisionTreeRegressor(max_depth=6, random_state=42)),("random_forest", RandomForestRegressor(n_estimators=100, random_state=42)),("gradient_boosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))]

def model_ranking_r2(models):
 models_score = [{"model" : "mean_baseline",
        "split" : "val",
        "MAE" : round(base_val_mae,4),
        "MSE" : round(base_val_mse,4),
        "RMSE" : round(base_val_rmse,4),
        "R2" : round(base_val_r2,4)
}]
 
 for model_name, model in models:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (model_name, model)
    ])
    pipeline.fit(X_train, y_train)
    val_pred = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    mse = mean_squared_error(y_val, val_pred)
    rmse = root_mean_squared_error(y_val, val_pred)
    r2_scr = r2_score(y_val, val_pred)
    models_score.append({
        "model" : model_name,
        "split" : "val",
        "MAE" : round(mae,4),
        "MSE" : round(mse,4),
        "RMSE" : round(rmse,4),
        "R2" : round(r2_scr,4)
    })
 return models_score



# 6 Learning-rate experiment.

gb_learning_rate = []
for lr in [0.01, 0.1, 1.0]:
    model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    model_gb.fit(X_train, y_train)
    pred = model_gb.predict(X_val)
    gb_learning_rate.append({
        "learning_rate" : lr,
        "val_rmse" : round(root_mean_squared_error(y_val, pred),4),
        "val_r2" : round(r2_score(y_val, pred),4)

    })
part_b_gb_learning_rate = pd.DataFrame(gb_learning_rate)
part_b_gb_learning_rate.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7 Cross-validated comparison and final model.
# Rank 1 linearregression, rank 2 gradientboosting (According to rmse)
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
b_cv_results = []
for model_name, model in [("linear_regression", LinearRegression()), ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))]:
   cv = cross_val_score(model, X_train_val, y_train_val, cv=kf, scoring="r2")
   b_cv_results.append({
      "model" : model_name,
      "mean_r2" : round(cv.mean(),4),
      "std_r2" : round(cv.std(),4)
                  })
part_b_metrics = pd.DataFrame(b_cv_results)
part_b_metrics.to_csv("outputs/part_b_cv.csv", index=False)

model_scores = model_ranking_r2(models)

best_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
best_model.fit(X_train_val, y_train_val)
y_pred_test = best_model.predict(X_test)
model_scores.append({
    "model": "gradient_boosting",
    "split": "test",
    "MAE": round(mean_absolute_error(y_test, y_pred_test), 4),
    "MSE": round(mean_squared_error(y_test, y_pred_test), 4),
    "RMSE": round(root_mean_squared_error(y_test, y_pred_test), 4),
    "R2": round(r2_score(y_test, y_pred_test), 4)
})

b_metrics = pd.DataFrame(model_scores)
b_metrics.to_csv("outputs/part_b_metrics.csv", index=False)