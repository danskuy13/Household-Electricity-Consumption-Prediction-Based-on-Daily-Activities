import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class FeatureEngineer:
    def __init__(self):
        pass

    def correlation_analysis(self, data):
        print("Performing correlation analysis...")

        # Hitung korelasi antar fitur numerik
        corr = data.corr(numeric_only=True)

        # Ambil korelasi terhadap target
        if 'konsumsi_listrik_kwh' in corr.columns:
            target_corr = corr['konsumsi_listrik_kwh'].drop('konsumsi_listrik_kwh')
        else:
            target_corr = pd.Series()

        # Buat folder untuk menyimpan grafik jika belum ada
        os.makedirs("reports/figures", exist_ok=True)

        # Plot heatmap korelasi
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matriks Korelasi Fitur")
        plt.tight_layout()

        # Simpan dan tampilkan heatmap
        plt.savefig("reports/figures/correlation_matrix.png")
        plt.show()
        plt.close()

        print("Saved and displayed correlation matrix heatmap.")
        return corr, target_corr

    def feature_importance_plot(self, feature_names, importances, model_name="Model"):
        print("Plotting feature importances...")
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feat_imp.values, y=feat_imp.index)
        plt.title(f"Feature Importances - {model_name}")
        plt.xlabel("Importance Score")
        plt.tight_layout()

        # Simpan dan tampilkan grafik
        fig_path = f"reports/figures/feature_importance_{model_name}.png"
        plt.savefig(fig_path)
        plt.show()
        plt.close()

        print(f"Feature importance plot saved to {fig_path} and displayed.")
