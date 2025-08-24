
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, confusion_matrix, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import joblib

RANDOM_STATE = 42
N_JOBS = -1

def load_data(path="diabetes.csv", target_col="Outcome"):
    df = pd.read_csv(path)
    # Treat impossible zeros as missing for certain columns (Pima dataset common fix)
    zero_as_missing = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for col in zero_as_missing:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return df, X, y

def build_preprocessor(X):
    num_features = X.columns.tolist()
    pre = ColumnTransformer([
        ("scale", StandardScaler(), num_features)
    ], remainder="drop")
    return pre

def get_models():
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced_subsample"),
        "gb": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "svc": SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
    }
    return models

def get_param_grids():
    grids = {
        "logreg": {"clf__C": [0.1, 1.0, 3.0]},
        "rf": {"clf__n_estimators": [250, 400], "clf__max_depth": [None, 6, 10]},
        "gb": {"clf__n_estimators": [150, 250], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]},
        "svc": {"clf__C": [0.5, 1.0, 2.0], "clf__gamma": ["scale", "auto"]},
    }
    return grids

def fit_and_select(X_train, y_train, pre):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = get_models()
    grids = get_param_grids()

    results = []
    best_estimators = {}

    for name, base in models.items():
        pipe = Pipeline([("pre", pre), ("clf", base)])
        grid = GridSearchCV(pipe, grids[name], scoring="roc_auc", cv=cv, n_jobs=N_JOBS, refit=True, verbose=0)
        grid.fit(X_train, y_train)
        auc = grid.best_score_
        results.append((name, auc, grid.best_params_))
        best_estimators[name] = grid.best_estimator_

    results.sort(key=lambda x: x[1], reverse=True)
    return results, best_estimators[results[0][0]]  # return leaderboard and top model

def tune_threshold(y_true, proba):
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    ix = np.argmax(j)
    return thr[ix]

def evaluate(model, X_valid, y_valid):
    proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, proba)
    th = tune_threshold(y_valid, proba)
    preds = (proba >= th).astype(int)

    acc = accuracy_score(y_valid, preds)
    cm = confusion_matrix(y_valid, preds)
    rpt = classification_report(y_valid, preds, digits=3)

    return {"auc": auc, "threshold": float(th), "accuracy": acc, "cm": cm, "report": rpt}

def calibrate_if_needed(model, X_train, y_train):
    # RandomForest/GradientBoosting often benefit from calibration
    needs_cal = any(k in model.named_steps["clf"].__class__.__name__.lower() for k in ["randomforest", "gradientboosting"])
    if needs_cal:
        base = model
        cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
        cal.fit(X_train, y_train)
        return cal
    return model

def train_main(csv_path="diabetes.csv"):
    df, X, y = load_data(csv_path)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pre = build_preprocessor(X)
    leaderboard, best = fit_and_select(X_train, y_train, pre)

    # optional calibration for tree ensembles
    best = calibrate_if_needed(best, X_train, y_train)

    metrics = evaluate(best, X_valid, y_valid)

    artifact = {"model": best, "features": X.columns.tolist(), "metrics": metrics, "leaderboard": leaderboard}
    joblib.dump(artifact, "diabetes_pipeline.joblib")

    print("Leaderboard (name, cv_auc, best_params):")
    for row in leaderboard:
        print(row)
    print("\nValidation metrics:", metrics)
    print("\nModel saved to diabetes_pipeline.joblib")

def predict_one(input_features, artifact_path="diabetes_pipeline.joblib"):
    art = joblib.load(artifact_path)
    model = art["model"]
    cols = art["features"]
    x = pd.DataFrame([input_features], columns=cols)
    proba = model.predict_proba(x)[:, 1][0]
    pred = int(proba >= art["metrics"]["threshold"])
    return pred, float(proba)

if __name__ == "__main__":
    # Run training only when executed directly
    try:
        train_main("diabetes.csv")
    except FileNotFoundError:
        print("Place diabetes.csv in the same folder to train. You can still use predict_one() after training.")
