import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

def preprocess_data():
    mlflow.set_experiment("Heart Disease - Preprocessing")
    with mlflow.start_run() as run:
        df = pd.read_csv("data/heart.csv")
        mlflow.set_tag("ml.step", "data_preprocessing")

        # Encode categorical: Sex, ChestPainType, etc.
        df_encoded = pd.get_dummies(df, drop_first=True)

        X = df_encoded.drop('HeartDisease', axis=1)
        y = df_encoded['HeartDisease']

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        os.makedirs("processed_data", exist_ok=True)
        pd.concat([X_train, y_train], axis=1).to_csv("processed_data/train.csv", index=False)
        pd.concat([X_test, y_test], axis=1).to_csv("processed_data/test.csv", index=False)

        mlflow.log_metric("train_rows", len(X_train))
        mlflow.log_metric("test_rows", len(X_test))
        mlflow.log_artifacts(os.path.abspath"processed_data", artifact_path="processed_data")

        print("Preprocessing done. Run ID:", run.info.run_id)

if __name__ == "__main__":
    preprocess_data()
