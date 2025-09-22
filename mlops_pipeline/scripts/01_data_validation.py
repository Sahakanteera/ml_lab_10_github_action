import pandas as pd
import mlflow

def validate_data():
    mlflow.set_experiment("Heart Disease - Data Validation")
    with mlflow.start_run():
        df = pd.read_csv("data/heart.csv")
        mlflow.set_tag("ml.step", "data_validation")

        num_rows, num_cols = df.shape
        missing_values = df.isnull().sum().sum()
        num_classes = df['HeartDisease'].nunique()

        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_param("num_classes", num_classes)

        status = "Success" if missing_values == 0 and num_classes == 2 else "Failed"
        mlflow.log_param("validation_status", status)

        print(f"Validation complete. Status: {status}")

if __name__ == "__main__":
    validate_data()
