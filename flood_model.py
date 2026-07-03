import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

np.random.seed(42)
n = 1000

annual_rainfall = np.random.uniform(500, 3000, n)
cloud_visibility = np.random.uniform(0.1, 10.0, n)
monsoon_rainfall = np.random.uniform(200, 2000, n)
pre_monsoon_rainfall = np.random.uniform(50, 500, n)
post_monsoon_rainfall = np.random.uniform(30, 400, n)
winter_rainfall = np.random.uniform(10, 200, n)

flood = (
    (annual_rainfall > 2000) |
    ((monsoon_rainfall > 1200) & (cloud_visibility < 3)) |
    ((annual_rainfall > 1500) & (monsoon_rainfall > 900))
).astype(int)

df = pd.DataFrame({
    "annual_rainfall": annual_rainfall,
    "cloud_visibility": cloud_visibility,
    "monsoon_rainfall": monsoon_rainfall,
    "pre_monsoon_rainfall": pre_monsoon_rainfall,
    "post_monsoon_rainfall": post_monsoon_rainfall,
    "winter_rainfall": winter_rainfall,
    "flood": flood
})

X = df.drop("flood", axis=1)
y = df["flood"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

best_model, best_acc, best_name = None, 0, ""

print("Model Performance:")
print("-" * 35)
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:<20} Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc, best_model, best_name = acc, model, name

print("-" * 35)
print(f"Best Model: {best_name} ({best_acc:.4f})")

with open("flood_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model saved as flood_model.pkl")

# --- Plot 1: Model Accuracy Comparison ---
names = list(models.keys())
accuracies = [accuracy_score(y_test, m.predict(X_test)) for m in models.values()]

plt.figure(figsize=(8, 5))
bars = plt.bar(names, accuracies, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
plt.ylim(0.8, 1.0)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
             f"{acc:.4f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("static/accuracy_comparison.png")
plt.close()
print("Saved: static/accuracy_comparison.png")

# --- Plot 2: Feature Importance (best model if supported) ---
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feat_names = X.columns.tolist()
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar([feat_names[i] for i in sorted_idx], importances[sorted_idx], color="#4C72B0")
    plt.ylabel("Importance")
    plt.title(f"Feature Importance ({best_name})")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("static/feature_importance.png")
    plt.close()
    print("Saved: static/feature_importance.png")

# --- Plot 3: Confusion Matrix (best model) ---
cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Flood", "Flood"],
            yticklabels=["No Flood", "Flood"])
plt.title(f"Confusion Matrix ({best_name})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
plt.close()
print("Saved: static/confusion_matrix.png")

print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test), target_names=["No Flood", "Flood"]))
