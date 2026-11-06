import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_classif
# 1 Stratified split. 
df = pd.read_csv("data/modelling_table.csv")
df.head()
X = df.drop(columns=["is_delayed", "delivery_hours"])
y = df["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 2 Blind baseline and three classifiers,
y_val_pred_baseline = np.full(shape=len(y_val), fill_value=0)
cm_baseline = confusion_matrix(y_val, y_val_pred_baseline)
baseline_accuracy = accuracy_score(y_val, y_val_pred_baseline)
precision_baseline = precision_score(y_val, y_val_pred_baseline)
recall_baseline = recall_score(y_val, y_val_pred_baseline)
f1_score_baseline = f1_score(y_val, y_val_pred_baseline)

c_models = [{
    "model" : "blind_all_on_time",
    "accuracy" : round(baseline_accuracy,4),
    "precision" : round(precision_baseline,4),
    "recall" : round(recall_baseline,4),
    "f1_score" : round(f1_score_baseline,4)
}]

cm = [{
    "model" : "blind_all_on_time",
    "TN" : cm_baseline[0][0],
    "FP" : cm_baseline[0][1],
    "FN" : cm_baseline[1][0],
    "TP" : cm_baseline[1][1]
}]

logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])
logistic_pipeline.fit(X_train, y_train)
logistic_val_pred = logistic_pipeline.predict(X_val)

c_models.append({
    "model" : "logistic_regression",
    "accuracy" : round(accuracy_score(y_val, logistic_val_pred),4),
    "precision" : round(precision_score(y_val, logistic_val_pred),4),
    "recall" : round(recall_score(y_val, logistic_val_pred),4),
    "f1_score" : round(f1_score(y_val, logistic_val_pred),4)
})
cm.append({
    "model" : "logistic_regression",
    "TN" : confusion_matrix(y_val, logistic_val_pred)[0][0],
    "FP" : confusion_matrix(y_val, logistic_val_pred)[0][1],
    "FN" : confusion_matrix(y_val, logistic_val_pred)[1][0],
    "TP" : confusion_matrix(y_val, logistic_val_pred)[1][1]
})

depts = [2, 3, 4, 5, 6, 8, 10]
f1_scores = []
for d in depts:
    model = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    f1_scores.append(f1_score(y_val, y_val_pred))

best_depth = depts[np.argmax(f1_scores)]

decision_tree_model = DecisionTreeClassifier(criterion="gini", max_depth=best_depth, random_state=42)
decision_tree_model.fit(X_train, y_train)
decision_tree_val_pred = decision_tree_model.predict(X_val)

c_models.append({
    "model" : f"decision_tree_depth_{best_depth}",
    "accuracy" : round(accuracy_score(y_val, decision_tree_val_pred),4),
    "precision" : round(precision_score(y_val, decision_tree_val_pred),4),
    "recall" : round(recall_score(y_val, decision_tree_val_pred),4),
    "f1_score" : round(f1_score(y_val, decision_tree_val_pred),4)
})
cm.append({
    "model" : f"decision_tree_depth_{best_depth}",
    "TN" : confusion_matrix(y_val, decision_tree_val_pred)[0][0],
    "FP" : confusion_matrix(y_val, decision_tree_val_pred)[0][1],
    "FN" : confusion_matrix(y_val, decision_tree_val_pred)[1][0],
    "TP" : confusion_matrix(y_val, decision_tree_val_pred)[1][1]
})

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
random_forest_val_pred = random_forest_model.predict(X_val)

c_models.append({
    "model" : "random_forest_default",
    "accuracy" : round(accuracy_score(y_val, random_forest_val_pred),4),
    "precision" : round(precision_score(y_val, random_forest_val_pred),4),
    "recall" : round(recall_score(y_val, random_forest_val_pred),4),
    "f1_score" : round(f1_score(y_val, random_forest_val_pred),4)
})
cm.append({
    "model" : "random_forest_default",
    "TN" : confusion_matrix(y_val, random_forest_val_pred)[0][0],
    "FP" : confusion_matrix(y_val, random_forest_val_pred)[0][1],
    "FN" : confusion_matrix(y_val, random_forest_val_pred)[1][0],
    "TP" : confusion_matrix(y_val, random_forest_val_pred)[1][1]
})

c_models = pd.merge(pd.DataFrame(c_models), pd.DataFrame(cm), on="model")

c_models.to_csv("outputs/part_c_models.csv", index=False)


# 3 Which columns carry signal?

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
feature_importances = model.feature_importances_

selector = SelectKBest(score_func=f_classif, k="all")
selector.fit(X_train, y_train)
f_score = selector.scores_

scores = {
    "feature" : X_train.columns,
    "f_score" : f_score.round(4),
    "rf_importance" :feature_importances.round(4)
}

scores_df = pd.DataFrame(scores).reset_index(drop=True)
scores_df["f_rank"] = pd.DataFrame(scores)["f_score"].rank(ascending=False, method="min").astype(int)
scores_df["is_noise"] = np.where(scores_df["f_rank"] >= 13, 1, 0)
score_df = scores_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
sorted_score_df = score_df.sort_values(by="f_rank", ascending=True).reset_index(drop=True)
sorted_score_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

c_selectK = []
for k in [5, 7, 10, 12,17]:
    rf_pipeline = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=k)),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    val_f1 = f1_score(y_val, rf_pipeline.predict(X_val))
    c_selectK.append((k, round(val_f1,4)))
c_select_df = pd.DataFrame(c_selectK, columns=["k", "val_f1"])
c_select_df.to_csv("outputs/part_c_selectk.csv", index=False)

# 4 Tuning
c_tuning = []
param_grid = {
    "n_estimators" : [50, 100, 150],
    "max_depth" : [4, 6, 8],
    "min_samples_leaf" : [1, 5, 10]
}
gd_candidates = len(param_grid["n_estimators"])*len(param_grid["max_depth"])*len(param_grid["min_samples_leaf"])

gd_search_cv = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5)
rd_search_cv = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42), param_distributions=param_grid, n_iter=9, cv=5, random_state=42)


models = [("GridSearchCV", gd_search_cv), ("RandomizedSearchCV", rd_search_cv)]
for model_name, model in models:
    model_pipeline = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=7)),
        ("model", model)])
    
    model_pipeline.fit(X_train, y_train)
    c_tuning.append({
        "search" : model_name,
        "candidates" : gd_candidates if model_name == "GridSearchCV" else 9,
        "cv_folds" : 5,
        "total_fits" : gd_candidates*5 if model_name == "GridSearchCV" else 9*5,
        "best_params" : model.best_params_,
        "best_cv_score" : round(model.best_score_,4),
        "val_f1" : round(f1_score(y_val, model_pipeline.predict(X_val)),4)
    })

default_forest_same_7_features_pipeline = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(n_estimators=100, min_samples_leaf=1, random_state=42))
])
default_forest_same_7_features_pipeline.fit(X_train, y_train)
c_tuning.append({
        "search" : "default_forest_same_7_features",
        "candidates" : 1,
        "cv_folds" : 0,
        "total_fits" : 1,
        "best_params" : "default",
        "best_cv_score" : [],
        "val_f1" : round(f1_score(y_val, default_forest_same_7_features_pipeline.predict(X_val)),4)
    })


c_tuning_df = pd.DataFrame(c_tuning)
c_tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)

# 5 Choosing the final configuration.
best_row_c_tuning_on_val_f1 = c_tuning_df.loc[c_tuning_df["val_f1"].idxmax()]


final_model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_depth=6, random_state=42))
])
final_model.fit(X_train, y_train)


# 6 Alert threshold.
final_model_val_pred_proba = final_model.predict_proba(X_val)[:,1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

cm_final = []
for t in thresholds:
    val_pred = np.where(final_model_val_pred_proba >= t, 1, 0)
    cm_final.append({
        "threshold" : t,
        "TN" : confusion_matrix(y_val, val_pred)[0][0],
        "FP" : confusion_matrix(y_val, val_pred)[0][1],
        "FN" : confusion_matrix(y_val, val_pred)[1][0],
        "TP" : confusion_matrix(y_val, val_pred)[1][1],
        "accuracy" : round(accuracy_score(y_val, val_pred),4),
        "precision" : round(precision_score(y_val, val_pred),4),
        "recall" : round(recall_score(y_val, val_pred),4),
        "f1" : round(f1_score(y_val, val_pred),4)
        })

c_thresholds = pd.DataFrame(cm_final)
c_thresholds.to_csv("outputs/part_c_thresholds.csv", index=False)

# 7 One test evaluation.
final_test = []
test_pred_proba = final_model.predict_proba(X_test)[:,1]
test_pred = np.where(test_pred_proba >= 0.5, 1, 0)
final_test.append({
    "TN" : confusion_matrix(y_test, test_pred)[0][0],
    "FP" : confusion_matrix(y_test, test_pred)[0][1],
    "FN" : confusion_matrix(y_test, test_pred)[1][0],
    "TP" : confusion_matrix(y_test, test_pred)[1][1],
    "accuracy" : round(accuracy_score(y_test, test_pred),4),
    "precision" : round(precision_score(y_test, test_pred),4),
    "recall" : round(recall_score(y_test, test_pred),4),
    "f1" : round(f1_score(y_test, test_pred),4)
})
final_test_df = pd.DataFrame(final_test)
final_test_df.to_csv("outputs/part_c_final_test.csv", index=False)

final_config = {
    "selector_k" : 7,
    "n_estimators" : 100,
    "max_depth" : 6,
    "min_samples_leaf" : 1,
    "threshold" : 0.5,
    "chosen_from" : "default_forest_same_7_features"
}
final_config_df = pd.DataFrame([final_config])
final_config_df.to_csv("outputs/part_c_final_config.csv", index=False)
