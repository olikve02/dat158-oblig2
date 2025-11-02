import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

# Column names from UCI (class + 22 categorical features)
COLS = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]

def load_data():
    df = pd.read_csv(DATA_URL, header=None, names=COLS)
    # Replace '?' with NaN for imputation
    df = df.replace("?", np.nan)
    return df

def main():
    df = load_data()
    # Target: 'e' = edible, 'p' = poisonous
    y = df["class"].map({"e": 0, "p": 1}).astype(int)
    X = df.drop(columns=["class"])

    # Keep track of original categories for UI selectboxes
    cat_levels = {c: sorted([v for v in X[c].dropna().unique()]) for c in X.columns}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    cat_features = list(X.columns)

    preproc = ColumnTransformer(
        transformers=[
            ("cat",
             Pipeline(steps=[
                 ("imputer", SimpleImputer(strategy="most_frequent")),
                 ("ohe", OneHotEncoder(handle_unknown="ignore"))
             ]),
             cat_features)
        ],
        remainder="drop"
    )

    model = Pipeline(steps=[
        ("preproc", preproc),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("F1 (poisonous=1):", f1)
    print(classification_report(y_test, y_pred, target_names=["edible", "poisonous"]))

    joblib.dump({
        "model": model,
        "columns": cat_features,
        "cat_levels": cat_levels,
        "target_names": ["edible", "poisonous"]
    }, "model.joblib")
    print("Saved model to model.joblib")

if __name__ == "__main__":
    main()
