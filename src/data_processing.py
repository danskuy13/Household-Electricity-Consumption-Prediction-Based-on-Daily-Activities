import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.feature_names = []

    def load_data(self, path):
        print(f"Loading data from: {path}")
        try:
            df = pd.read_csv(path)
            print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def process_pipeline(self, df):
        print("Processing pipeline started...")
        df = df.copy()

        # Pisahkan target
        X = df.drop("konsumsi_listrik_kwh", axis=1)
        y = df["konsumsi_listrik_kwh"]

        # Identifikasi kolom kategorikal dan numerikal
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Pipeline transformasi
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ])

        pipeline = Pipeline([
            ("preprocessor", preprocessor)
        ])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit transform ke data latih, transform ke data uji
        X_train_scaled = pipeline.fit_transform(X_train)
        X_test_scaled = pipeline.transform(X_test)

        # Dapatkan feature names hasil transformasi
        ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = ohe.get_feature_names_out(categorical_cols)
        all_feature_names = numerical_cols + cat_feature_names.tolist()
        self.feature_names = all_feature_names

        self.scaler = pipeline
        print("Data preprocessing completed.")

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test

    def generate_summary_report(self, df, results_df, final_metrics):
        report_path = "reports/final_report.md"

        summary = f"""# Laporan Proyek - Prediksi Konsumsi Listrik Rumah Tangga

## Ringkasan Dataset
- Jumlah Sampel: {df.shape[0]}
- Jumlah Fitur: {df.shape[1] - 1}
- Target: konsumsi_listrik_kwh

## Statistik Data:
{df.describe().to_markdown()}

## Hasil Evaluasi Model
{results_df.to_markdown()}

## Model Terbaik: {final_metrics['Model']}

| Metric | Value |
|--------|-------|
| R²     | {final_metrics['R²']:.4f} |
| RMSE   | {final_metrics['RMSE']:.4f} |
| MAE    | {final_metrics['MAE']:.4f} |
| MAPE (%) | {final_metrics['MAPE (%)']:.2f} |

---
Laporan ini dibuat secara otomatis oleh pipeline.
"""

        with open(report_path, "w", encoding='utf-8') as f:
            f.write(summary)
        print(f"Final report saved to {report_path}")
