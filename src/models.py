from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

class ElectricityModels:
    def __init__(self):
        self.best_model = None
        self.trained_models = {}

    def initialize_models(self):
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42)
        }
        return models

    def train_models(self, models, X_train, y_train):
        trained = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained[name] = model
        self.trained_models = trained
        return trained

    def evaluate_models(self, models, X_train, y_train):
        print("Evaluating models with cross-validation...")
        results = {}
        for name, model in models.items():
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
            results[name] = {
                "CV R² Mean": scores.mean(),
                "CV R² Std": scores.std()
            }
        df_results = pd.DataFrame(results).T.sort_values("CV R² Mean", ascending=False)
        return df_results

    def plot_model_comparison(self, results_df):
        print("Plotting model comparison...")
        plt.figure(figsize=(8, 5))
        sns.barplot(x=results_df.index, y="CV R² Mean", data=results_df.reset_index())
        plt.title("Perbandingan R² Antar Model")
        plt.ylabel("Mean CV R²")
        plt.xlabel("Model")
        plt.xticks(rotation=15)
        plt.tight_layout()
        os.makedirs("reports/figures", exist_ok=True)
        plt.savefig("reports/figures/model_comparison.png")
        plt.show()  # <<--- Agar langsung tampil saat main.py dijalankan
        print("Saved and displayed model comparison plot.")

    def hyperparameter_tuning(self, X_train, y_train):
        print("Tuning Random Forest...")
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20]
        }
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='r2')
        grid.fit(X_train, y_train)
        print(f"Best Parameters: {grid.best_params_}")
        self.best_model = ("Random Forest", grid.best_estimator_)
        return self.best_model[0], self.best_model[1], grid.best_params_

    def get_feature_importance(self, feature_names):
        if self.best_model is None:
            print("No model trained.")
            return None

        model_name, model = self.best_model
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            if len(importances) != len(feature_names):
                print(f"Warning: Panjang feature_names ({len(feature_names)}) tidak cocok dengan importances ({len(importances)})")
                return None
            return pd.Series(importances, index=feature_names)
        else:
            print(f"Model {model_name} tidak mendukung feature importance.")
            return None

    def save_models(self):
        if self.best_model is None:
            print("No model to save.")
            return
        model_name, model = self.best_model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, f"models/{model_name.replace(' ', '_')}_model.pkl")
        print(f"Saved best model: {model_name}")
