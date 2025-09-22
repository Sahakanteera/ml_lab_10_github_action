import mlflow
import pandas as pd
import os

def load_and_predict():
    # เพิ่มบรรทัดนี้
    if not os.environ.get('MLFLOW_TRACKING_URI'):
        mlflow.set_tracking_uri("file:./mlruns")
    
    try:
        model = mlflow.pyfunc.load_model("models:/heart-classifier/Staging")
    except Exception as e:
        print(f"Warning: Could not load model from registry: {e}")
        print("This might be expected in CI/CD environments")
        return

    df = pd.read_csv("data/heart.csv")
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop("HeartDisease", axis=1)
    sample = X.iloc[[0]]

    pred = model.predict(sample)
    print(f"Predicted: {pred[0]} (Actual: {df.HeartDisease.iloc[0]})")

if __name__ == "__main__":
        load_and_predict()