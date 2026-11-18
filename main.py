"""
main.py
-------
Runs the full pipeline described in the synopsis Methodology:

1. Load & preprocess the disease dataset               (data.py)
2. Split into simulated hospital clients (IID / non-IID) (data.py)
3. Train a Federated Learning global model via FedAvg    (federated.py)
4. Train a centralized baseline on the same data          (centralized.py)
5. Evaluate both with Accuracy / Precision / Recall / F1 / ROC-AUC (utils.py)
6. Save comparison plots + a results table to results/

Usage:
    python3 main.py                     # IID, 5 hospitals (default)
    python3 main.py --hospitals 8 --non_iid --rounds 30
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import load_dataset, split_into_hospitals
from federated import run_federated_training
from centralized import run_centralized_training
from utils import evaluate_model, print_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    parser = argparse.ArgumentParser(description="Federated Learning for Disease Prediction")
    parser.add_argument("--hospitals", type=int, default=5, help="number of simulated hospitals")
    parser.add_argument("--rounds", type=int, default=25, help="federated communication rounds")
    parser.add_argument("--local_epochs", type=int, default=3, help="local epochs per round per hospital")
    parser.add_argument("--non_iid", action="store_true", help="use skewed non-IID hospital splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Federated Learning for Disease Prediction")
    print(f"Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print(f"Hospitals: {args.hospitals} | IID: {not args.non_iid} | Rounds: {args.rounds}")
    print("=" * 60)

    # 1. Load data
    X_train, X_test, y_train, y_test, feature_names = load_dataset(seed=args.seed)
    n_features = X_train.shape[1]

    # 2. Split into simulated hospitals
    hospital_data = split_into_hospitals(
        X_train, y_train, n_hospitals=args.hospitals, iid=not args.non_iid, seed=args.seed
    )
    print("\nHospital data distribution:")
    for i, (Xh, yh) in enumerate(hospital_data):
        print(f"  Hospital {i+1}: {len(Xh)} samples | malignant={int((yh==0).sum())} benign={int((yh==1).sum())}")

    # 3. Federated training
    print("\nRunning Federated Learning (FedAvg)...")
    fl_model, fl_history = run_federated_training(
        hospital_data, X_test, y_test, n_features,
        n_rounds=args.rounds, local_epochs=args.local_epochs, seed=args.seed,
    )

    # 4. Centralized baseline
    print("Running centralized baseline...")
    # match total gradient steps roughly: rounds * local_epochs epochs over pooled data
    central_epochs = args.rounds * args.local_epochs
    central_model, central_history = run_centralized_training(
        X_train, y_train, X_test, y_test, n_features,
        epochs=central_epochs, seed=args.seed,
    )

    # 5. Final evaluation
    fl_metrics = evaluate_model(fl_model, X_test, y_test)
    central_metrics = evaluate_model(central_model, X_test, y_test)
    print_metrics("Federated Learning (FedAvg)", fl_metrics)
    print_metrics("Centralized ML", central_metrics)

    # 6. Save results
    with open(os.path.join(RESULTS_DIR, "metrics_comparison.json"), "w") as f:
        json.dump({"federated": fl_metrics, "centralized": central_metrics}, f, indent=2)

    # Plot 1: accuracy over communication rounds vs centralized epochs
    plt.figure(figsize=(7, 5))
    plt.plot([h["round"] for h in fl_history], [h["test_accuracy"] for h in fl_history], label="Federated (per round)")
    plt.plot([h["epoch"] / args.local_epochs for h in central_history],
              [h["test_accuracy"] for h in central_history],
              label="Centralized (per equiv. round)", alpha=0.7)
    plt.xlabel("Communication Round (equivalent)")
    plt.ylabel("Test Accuracy")
    plt.title("Federated vs Centralized: Accuracy over Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"), dpi=150)
    plt.close()

    # Plot 2: final metrics bar comparison
    labels = list(fl_metrics.keys())
    fl_vals = [fl_metrics[k] for k in labels]
    c_vals = [central_metrics[k] for k in labels]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, fl_vals, width, label="Federated (FedAvg)")
    plt.bar(x + width/2, c_vals, width, label="Centralized")
    plt.xticks(x, labels, rotation=20)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Final Model Comparison: Federated vs Centralized")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_bar_comparison.png"), dpi=150)
    plt.close()

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(" - accuracy_curve.png")
    print(" - metrics_bar_comparison.png")
    print(" - metrics_comparison.json")


if __name__ == "__main__":
    main()
