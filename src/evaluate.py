import joblib
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import load_data, preprocess_data, split_data

model = joblib.load("models/churn_model.pkl")

df = load_data("data/raw/churn.csv")
X, y, _ = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
