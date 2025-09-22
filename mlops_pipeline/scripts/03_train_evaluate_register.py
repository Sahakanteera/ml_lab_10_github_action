import pandas as pd
import sys
import os
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# ลบบรรทัดนี้ออก: from mlflow.artifacts import download_artifacts

def train(preprocessing_run_id):
    # เพิ่มบรรทัดนี้
    if not os.environ.get('MLFLOW_TRACKING_URI'):
        mlflow.set_tracking_uri("file:./mlruns")
        
    mlflow.set_experiment("Heart Disease - Train")
    with mlflow.start_run():
        mlflow.set_tag("ml.step", "training")

        # แก้ส่วนนี้ - อ่านจาก processed_data ตรงๆ แทน download_artifacts
        try:
            train_df = pd.read_csv("processed_data/train.csv")
            test_df = pd.read_csv("processed_data/test.csv")
        except FileNotFoundError:
            print("Error: processed_data files not found. Please run preprocessing first.")
            return

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
            try:
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/heart_model"
                mlflow.register_model(model_uri, "heart-classifier")
                print("Model registered.")
            except Exception as e:
                print(f"Warning: Could not register model: {e}")

        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else ""
    train(run_id)
    