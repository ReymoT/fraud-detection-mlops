import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, classification_report

from src.config import DATA_PATH, MODEL_DIR, FEATURE_COLS, TOP_PERCENTILE, EXPERIMENT_NAME
from src.ingest import load_data
from src.features import build_features
from src.evaluate import evaluate_model

import mlflow
import mlflow.xgboost

def main():
    os.makedirs(MODEL_DIR, exist_ok = True)

    df = load_data(DATA_PATH)
    df = build_features(df)

    new_df = df[FEATURE_COLS + ["is_fraud"]].dropna()

    new_df = pd.get_dummies(
        new_df,
        columns = ["category", "gender", "state"]
    )

    new_df = new_df.sort_values("unix_time")

    train_size = int(0.8 * len(new_df))
    train_df = new_df.iloc[:train_size].copy()
    test_df = new_df.iloc[train_size:].copy()

    freq = train_df["merchant"].value_counts(normalize = True)

    train_df["merchant_freq"] = train_df["merchant"].map(freq)
    test_df["merchant_freq"] = test_df["merchant"].map(freq).fillna(0)

    train_df = train_df.drop(["merchant", "unix_time"], axis = 1)
    test_df = test_df.drop(["merchant", "unix_time"], axis = 1)

    X_train = train_df.drop("is_fraud", axis = 1)
    y_train = train_df["is_fraud"]

    X_test = test_df.drop("is_fraud", axis = 1)
    y_test = test_df["is_fraud"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "top_percentile": TOP_PERCENTILE,
        }

        mlflow.log_params(params)

        model = XGBClassifier(
            **params,
            random_state=42
        )

        model.fit(X_train, y_train)

        metrics, y_scores, top_k = evaluate_model(
            model,
            X_test,
            y_test,
            top_percentile=TOP_PERCENTILE
        )

        mlflow.log_metric("pr_auc", metrics["pr_auc"])
        mlflow.log_metric("precision_at_0_5_percent", metrics["precision_at_k"])
        mlflow.log_metric("threshold", metrics["threshold"])
        mlflow.log_metric("recall_at_kt", metrics["recall_at_k"])
        mlflow.log_metric("baseline_fraud_rate", metrics["baseline_fraud_rate"])

        print("PR-AUC:", metrics["pr_auc"])
        print("Threshold:", metrics["threshold"])
        print("Precision at 0.5%:", metrics["precision_at_k"])
        print(metrics["classification_report"])

        joblib.dump(model, f"{MODEL_DIR}/fraud_model.pkl")
        joblib.dump(freq.to_dict(), f"{MODEL_DIR}/merchant_freq.pkl")
        joblib.dump(list(X_train.columns), f"{MODEL_DIR}/model_columns.pkl")
        joblib.dump(metrics["threshold"], f"{MODEL_DIR}/threshold.pkl")

        mlflow.xgboost.log_model(model, name = "xgboost_fraud_model")
        mlflow.register_model(
            "runs:/{}/xgboost_fraud_model".format(mlflow.active_run().info.run_id),
            "fraud_model"
        )

        mlflow.log_artifact(f"{MODEL_DIR}/merchant_freq.pkl")
        mlflow.log_artifact(f"{MODEL_DIR}/model_columns.pkl")
        mlflow.log_artifact(f"{MODEL_DIR}/threshold.pkl")
        with open("report.txt", "w") as f:
            f.write(metrics["classification_report"])

        mlflow.log_artifact("report.txt")

        importances = model.feature_importances_
        features = X_train.columns

        feat_df = pd.DataFrame({
            "feature": features,
            "importance": importances
        })

        feat_df = feat_df.sort_values("importance", ascending = False).head(15)

        plt.figure(figsize = (8, 6))
        plt.barh(feat_df["feature"], feat_df["importance"])
        plt.gca().invert_yaxis()
        plt.title("Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig("feature_importance.png")

        mlflow.log_artifact("feature_importance.png")

if __name__ == "__main__":
    main()