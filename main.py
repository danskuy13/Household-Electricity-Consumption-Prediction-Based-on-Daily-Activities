"""
Prediksi Konsumsi Listrik
Author: [Dani Ramdani]
Date: 26-06-2025
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.models import ElectricityModels
from src.evaluation import ModelEvaluator

def create_directories():
    directories = [
        'data/raw', 'data/processed', 'data/generated',
        'models', 'reports/figures'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")

def main():
    print("="*60)
    print("PREDIKSI KONSUMSI LISTRIK RUMAH TANGGA")
    print("="*60)

    # 1. Create directories
    create_directories()

    # 2. Load dataset
    dataset_path = 'data/generated/listrik_rumah.csv'
    if not os.path.exists(dataset_path):
        print("Dataset belum tersedia, mencoba generate...")
        try:
            from generate_dataset import main as generate_data
            generate_data()
        except Exception as e:
            print(f"Gagal generate dataset: {e}")
            return

    print("\n1. DATA PROCESSING...")
    processor = DataProcessor()
    data = processor.load_data(dataset_path)
    X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = processor.process_pipeline(data)
    print("Data loaded dan diproses dengan sukses")

    # 3. Feature engineering
    print("\n2. FEATURE ENGINEERING...")
    engineer = FeatureEngineer()
    corr, target_corr = engineer.correlation_analysis(data)

    # 4. Model training & evaluation
    print("\n3. MODEL TRAINING & EVALUATION...")
    modeler = ElectricityModels()
    models = modeler.initialize_models()
    trained_models = modeler.train_models(models, X_train_scaled, y_train)
    results_df = modeler.evaluate_models(trained_models, X_train_scaled, y_train)
    print(results_df)
    modeler.plot_model_comparison(results_df)

    # 5. Hyperparameter tuning
    print("\n4. HYPERPARAMETER TUNING...")
    best_model_name, best_model, best_params = modeler.hyperparameter_tuning(X_train_scaled, y_train)
    modeler.best_model = (best_model_name, best_model)

    # 6. Final evaluation
    print("\n5. FINAL EVALUATION...")
    evaluator = ModelEvaluator()
    y_pred = best_model.predict(X_test_scaled)
    final_metrics = evaluator.calculate_metrics(y_test, y_pred, best_model_name)
    print("Final metrics:")
    print(final_metrics)
    evaluator.generate_evaluation_report(best_model_name, y_test, y_pred, best_model, X_test, processor.feature_names)
    evaluator.plot_learning_curves(best_model, X_train_scaled, y_train, best_model_name)

    # 7. Save model
    print("\n6. SAVE MODEL...")
    modeler.save_models()
    if processor.scaler:
        joblib.dump(processor.scaler, 'models/scaler.pkl')

    # 8. Optional: plot feature importance
    feat_imp = modeler.get_feature_importance(processor.feature_names)
    if feat_imp is not None:
       engineer.feature_importance_plot(processor.feature_names, feat_imp.values, best_model_name)

    print("\nSEMUA PROSES SELESAI!")

if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    sns.set_palette("husl")
    main()
