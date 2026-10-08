#%%
import sys
from config import PROJECT_ROOT

p = PROJECT_ROOT / "lab1"
sys.path.append(str(p))

import numpy as np
from preprocessing import get_raw_data, process_features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

def main():
    X, y = get_raw_data()

    models = {
        "KNeighborsClassifier": KNeighborsClassifier()
    }

    metrics = {
        "Accuracy": accuracy_score,
        "Balanced Accuracy": balanced_accuracy_score,
        "Precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "Recall": recall_score,
        "F1-Score": f1_score
    }

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    results = {}
    for name in models:
        results[name] = {}
        for metric in metrics:
            results[name][metric] = []

    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = process_features(X.iloc[train_idx], X.iloc[test_idx])
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            for metric_name, metric_fn in metrics.items():
                score = metric_fn(y_test, y_pred)
                results[name][metric_name].append(score)

    for name in models:
        print(f"\n{name}:")
        print("-" * 50)
        for metric_name in metrics:
            scores = results[name][metric_name]
            mean = np.mean(scores)
            print(f"{metric_name}: {mean:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
