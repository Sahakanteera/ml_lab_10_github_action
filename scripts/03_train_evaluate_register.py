import pandas as pd
import sys
import os
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.artifacts import download_artifacts

def train(preprocessing_run_id):
    mlflow.set_experiment("Heart Disease - Train")
    with mlflow.start_run():
        mlflow.set_tag("ml.step", "training")

        path = download_artifacts(run_id=preprocessing_run_id, artifact_path="processed_data")
        train_df = pd.read_csv(os.path.join(path, "train.csv"))
        test_df = pd.read_csv(os.path.join(path, "test.csv"))

        X_train = train_df.drop("HeartDisease", axis=1)
        y_train = train_df["HeartDisease"]
        X_test = test_df.drop("HeartDisease", axis=1)
        y_test = test_df["HeartDisease"]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(pipe, "heart_model")

        if acc >= 0.85:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/heart_model"
            mlflow.register_model(model_uri, "heart-classifier")
            print("Model registered.")

        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else ""
    train(run_id)
