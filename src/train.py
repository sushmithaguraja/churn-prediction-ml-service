import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from data_preprocessing import load_data, preprocess_data, split_data

DATA_PATH = "churn.csv"

df = load_data(DATA_PATH)
X, y, preprocessor = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100)
}

best_model = None
best_score = 0

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)

    print(f"{name} accuracy: {score}")

    if score > best_score:
        best_score = score
        best_model = pipe

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/churn_model.pkl")
print("Model saved.")
