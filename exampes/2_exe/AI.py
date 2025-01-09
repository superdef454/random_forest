import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_data(file_path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    return data


def train_model(
    # data: pd.DataFrame,
    # target_column,
    # test_size,
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators,
    criterion,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    max_features,
    bootstrap,
    oob_score,
    n_jobs,
):
    # # Деление данных на признаки и целевую переменную
    # X = data.drop(columns=[target_column])
    # y = data[target_column]
    
    # # Преобразование категориальных данных в числовые
    # X = pd.get_dummies(X)

    # # Разделение данных на обучающую и тестовую выборки
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    if max_features == "None":
        max_features = None

    # Инициализация модели
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth if max_depth > 0 else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        oob_score=oob_score,
        n_jobs=n_jobs,
    )

    # Обучение модели
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time

    # Оценка качества модели
    accuracy = clf.score(X_test, y_test)

    # Важность признаков
    feature_importances = clf.feature_importances_

    # Сохранение модели
    model_filename = "random_forest_model.joblib"
    joblib.dump(clf, model_filename)
    model_size_kb = os.path.getsize(model_filename) / 1024

    return {
        'model': clf,
        'accuracy': accuracy,
        'train_time': train_time,
        'feature_importances': feature_importances,
        'model_size_kb': model_size_kb,
    }
