import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

class ModelEvaluator:
    def __init__(self):
        pass

    def calculate_metrics(self, y_true, y_pred, model_name):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        metrics = {
            "Model": model_name,
            "R²": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE (%)": mape
        }
        return metrics

    def generate_evaluation_report(self, model_name, y_true, y_pred, model, X_test, feature_names):
        os.makedirs("reports", exist_ok=True)
        os.makedirs("reports/figures", exist_ok=True)

        # Plot actual vs predicted
        plt.figure(figsize=(8, 5))
        plt.scatter(y_true, y_pred, alpha=0.7, color='royalblue')
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel("Actual Consumption")
        plt.ylabel("Predicted Consumption")
        plt.title(f"Actual vs Predicted - {model_name}")
        plt.tight_layout()
        plt.savefig(f"reports/figures/actual_vs_predicted_{model_name}.png")
        plt.show()  # <-- Tambahkan agar tampil di layar
        plt.close()
        print(f"Saved and displayed actual vs predicted plot for {model_name}")

    def plot_learning_curves(self, model, X_train, y_train, model_name):
        from sklearn.model_selection import learning_curve

        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42
        )

        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)

        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_mean, 'o-', label="Training Score")
        plt.plot(train_sizes, val_mean, 'o-', label="Validation Score")
        plt.xlabel("Training Set Size")
        plt.ylabel("R² Score")
        plt.title(f"Learning Curve - {model_name}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"reports/figures/learning_curve_{model_name}.png")
        plt.show()  # <-- Tambahkan agar tampil di layar
        plt.close()
        print(f"Saved and displayed learning curve for {model_name}")
