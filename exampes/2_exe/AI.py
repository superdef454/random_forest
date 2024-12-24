import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_data(file_path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    return data


def train_model(data, target_column, test_size, n_estimators, criterion, max_depth):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth if max_depth > 0 else None
    )

    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time

    accuracy = clf.score(X_test, y_test)

    feature_importances = clf.feature_importances_

    model_filename = "random_forest_model.joblib"
    joblib.dump(clf, model_filename)
    model_size_kb = os.path.getsize(model_filename) / 1024

    return {
        'model': clf,
        'accuracy': accuracy,
        'train_time': train_time,
        'feature_importances': feature_importances,
        'model_size_kb': model_size_kb
    }