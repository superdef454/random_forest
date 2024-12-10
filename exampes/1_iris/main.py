# Импорт необходимых библиотек
from sklearn.datasets import load_iris  # Для загрузки набора данных Iris
from sklearn.ensemble import RandomForestClassifier  # Для создания модели случайного леса
from sklearn.metrics import accuracy_score  # Для оценки точности модели
from sklearn.model_selection import train_test_split  # Для разделения данных на обучающую и тестовую выборки

# В этом скрипте мы будем использовать библиотеку scikit-learn для создания модели машинного обучения,
# которая будет распознавать виды ирисов на основе их характеристик.


if __name__ == "__main__":
    # Загрузка данных из набора данных Iris
    data = load_iris()
    X = data.data  # Признаки (характеристики) цветов
    y = data.target  # Метки классов (виды цветов)

    # Вывод общего количества данных
    print(f"Общее количество данных: {len(X)}")

    # Пример данных
    print(f"Пример данных: {X[0]}")  # Пример данных: [5.1 3.5 1.4 0.2]
    # Первое число - длина чашелистика
    # Второе число - ширина чашелистика
    # Третье число - длина лепестка
    # Четвертое число - ширина лепестка

    print(f"Пример метки класса: {y[0]}")  # Пример метки класса: 0
    # 0 - Iris Setosa
    # 1 - Iris Versicolour
    # 2 - Iris Virginica

    # Разделение данных на обучающую и тестовую выборки
    # test_size=0.3 означает, что 30% данных будут использованы для тестирования, чтобы иметь достаточно данных для оценки модели
    # random_state=42 фиксирует случайное разбиение для воспроизводимости результатов
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Вывод количества данных в обучающей и тестовой выборках
    print(f"Количество данных в обучающей выборке: {len(X_train)}")
    print(f"Количество данных в тестовой выборке: {len(X_test)}")

    # Создание модели случайного леса
    # n_estimators=100 означает, что будет создано 100 деревьев решений, что обычно достаточно для хорошей производительности модели
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Обучение модели на обучающей выборке
    clf.fit(X_train, y_train)

    # Предсказание меток классов на тестовой выборке
    y_pred = clf.predict(X_test)

    # Оценка точности модели
    # accuracy_score сравнивает предсказанные метки с истинными метками
    accuracy = accuracy_score(y_test, y_pred)

    # Вывод точности модели
    print(f"Точность модели: {accuracy:.2f}")


    # Визуализация важности признаков в отдельном окне
    
    # Импорт необходимых библиотек для визуализации
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Получение значимости признаков
    feature_importances = clf.feature_importances_
    features = data.feature_names
    features_ru = ["Длина чашелистика", "Ширина чашелистика", "Длина лепестка", "Ширина лепестка"]


    # Создание DataFrame для удобства визуализации
    fi_df = pd.DataFrame({'Признак': features_ru, 'Важность %': feature_importances})

    # Сортировка признаков по важности
    fi_df = fi_df.sort_values(by='Важность %', ascending=False)

    # Построение графика
    plt.figure(figsize=(10, 6))  # Размер графика
    sns.barplot(x='Важность %', y='Признак', data=fi_df)  # Построение графика
    plt.title('Важность признаков в модели случайного леса')  # Заголовок графика
    plt.show()  # Отображение графика
