"""
Step 4, 5 & 6: Model Building, Training/Testing, Evaluation
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")

df = pd.read_csv("/home/claude/student_perf/data/student_performance_cleaned.csv")

FEATURES = ["Study_Hours", "Attendance", "Previous_Score", "Assignments_Score",
            "Gender", "Parent_Education"]
TARGET = "Final_Marks"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (mainly benefits Linear Regression; tree models are scale-invariant
# but we keep a single consistent pipeline for simplicity)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
}

results = []
predictions = {}

for name, model in models.items():
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2})
    predictions[name] = preds

    joblib.dump(model, f"/home/claude/student_perf/models/{name.replace(' ', '_').lower()}.pkl")

results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
results_df.to_csv("/home/claude/student_perf/models/model_comparison.csv", index=False)
joblib.dump(scaler, "/home/claude/student_perf/models/scaler.pkl")

print("\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model: {best_model_name}")

# ---- Feature importance (Random Forest) ----
rf = models["Random Forest Regressor"]
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

plt.figure(figsize=(7, 5))
sns.barplot(x=importances.values, y=importances.index, color="steelblue")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("/home/claude/student_perf/plots/07_feature_importance.png", dpi=120)
plt.close()

# ---- Predicted vs Actual plots for each model ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, preds) in zip(axes, predictions.items()):
    ax.scatter(y_test, preds, alpha=0.6)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--", label="Ideal")
    ax.set_xlabel("Actual Marks")
    ax.set_ylabel("Predicted Marks")
    ax.set_title(name)
    ax.legend()
plt.tight_layout()
plt.savefig("/home/claude/student_perf/plots/08_predicted_vs_actual.png", dpi=120)
plt.close()

# ---- Model comparison bar chart ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, metric in zip(axes, ["MAE", "RMSE", "R2 Score"]):
    sns.barplot(data=results_df, x="Model", y=metric, ax=ax, palette="viridis")
    ax.set_title(metric)
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig("/home/claude/student_perf/plots/09_model_comparison.png", dpi=120)
plt.close()

print("\nAll plots and models saved.")
