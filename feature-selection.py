import argparse
import csv
import os
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature Selection using Perceptron")

    parser.add_argument("--accuracy-threshold", type=float, default=0.9,
                        help="Threshold for accuracy to consider features for selection")
    parser.add_argument("--capture-figures", action='store_true', default=False,
                        help="Capture the figures for each feature combination.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    org_data = load_breast_cancer()
    target = pd.Series(org_data.target, name="target")
    data = pd.DataFrame(org_data.data, columns=org_data.feature_names)
    data.index = data.index.astype(str)
    y = target
    X = data

    os.makedirs("results", exist_ok=True)

    results_file = "results/feature_combinations.csv"
    results = []

    with open(results_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Feature 1", "Feature 2", "Accuracy"])

    # Iterate over all combinations of 2 features
    combinations_list = list(combinations(X.columns, 2))
    for feature1, feature2 in tqdm(combinations_list, desc="Processing combinations"):
        X_subset = X[[feature1, feature2]]
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.2, random_state=42)

        # Train Perceptron model
        model = Perceptron()
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Plot data points
        if args.capture_figures:
            os.makedirs("figures/selection", exist_ok=True)
            plt.figure()
            plt.scatter(X_train[feature1], X_train[feature2],
                        c=y_train, cmap="viridis", label="Train")
            plt.scatter(X_test[feature1], X_test[feature2], c=y_test,
                        cmap="cool", marker="x", label="Test")
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.title(f"Feature Combination: {feature1} vs {feature2}")
            plt.legend()
            plt.savefig(f"figures/selection/{feature1}_vs_{feature2}.png")
            plt.close()

        results.append((feature1, feature2, accuracy))

    results.sort(key=lambda x: x[2], reverse=True)

    with open(results_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(results)

    features = []
    for feature1, feature2, _ in list(filter(lambda x: x[2] >= args.accuracy_threshold, results)):
        if feature1 not in features:
            features.append(feature1)

        if feature2 not in features:
            features.append(feature2)

    X = X[features]
    df = pd.DataFrame(X, columns=features)
    df["target"] = y.values
    df.set_index("target", inplace=True, drop=True)

    df.to_csv(
        f"results/selected_features_{args.accuracy_threshold}.csv", index=True)
