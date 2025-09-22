import mlflow
import pandas as pd

def load_and_predict():
    model = mlflow.pyfunc.load_model("models:/heart-classifier/Staging")

    df = pd.read_csv("data/heart.csv")
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop("HeartDisease", axis=1)
    sample = X.iloc[[0]]

    pred = model.predict(sample)
    print(f"Predicted: {pred[0]} (Actual: {df.HeartDisease.iloc[0]})")

if __name__ == "__main__":
    load_and_predict()
