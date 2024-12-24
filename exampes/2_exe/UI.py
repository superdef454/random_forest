import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from AI import load_data, train_model

default_font = "exampes/2_exe/fonts/Comfortaa-Medium.ttf"

def select_file_for_training_the_model():
    """Открывает диалоговое окно для выбора файла с данными."""
    def callback(sender, app_data):
        file_path = list(app_data['selections'].values())[0]
        file_name = list(app_data['selections'].keys())[0]
        dpg.set_value("file_name", file_name)
        data = load_data(file_path)
        dpg.set_item_user_data("target_column_combo", data)
        update_data_table()
        columns = list(data.columns)
        dpg.configure_item("target_column_combo", items=columns)
        add_log(f"Загружен файл: {file_path}", color=[0, 255, 0])
    with dpg.file_dialog(directory_selector=False,
                         show=True,
                         callback=callback,
                         width=500,
                         height=400,
                         ):
        # dpg.add_file_extension(".*", color=(255, 255, 255, 255))
        dpg.add_file_extension(".csv", color=(0, 255, 255, 255))

def choise_callback(sender, app_data, user_data):
    if not app_data or not user_data:
        return

    # Берём «последнюю» инструкцию сортировки — она соответствует
    # колонке, по которой пользователь только что кликнул.
    last_sort = app_data[-1]       # (column_id, is_ascending)
    column_id = last_sort[0]       # ID колонки
    is_ascending = last_sort[1]    # True/False — сортировка вверх или вниз

    # Из user_data таблицы достаём имя столбца
    col_name = user_data.get(column_id, "Unknown")

    # Устанавливаем выбранное имя столбца в целевую комбо
    dpg.set_value("target_column_combo", col_name)

def update_data_table():
    # Забираем DataFrame, сохранённый как пользовательские данные виджета "target_column_combo"
    data = dpg.get_item_user_data("target_column_combo")  # type: pd.DataFrame
    
    # Если данных нет, выходим из функции
    if data is None or data.empty:
        return

    # Удаляем все предыдущие колонки и строки в таблице "data_table"
    dpg.delete_item("data_table", children_only=True)

    # Создадим словарь для user_data таблицы: 
    # {column_id: column_name, ...}
    table_col_map = {}

    # Добавляем новые колонки
    for col_name in data.columns:
        # Создаём колонку
        col_id = dpg.add_table_column(
            label=col_name,
            parent="data_table",
            width_fixed=True,
            width=50,
        )
        # Запоминаем соответствие ID колонки → имя столбца
        table_col_map[col_id] = col_name

    # Добавляем строки (первые 3)
    for row_index in range(min(len(data), 3)):
        with dpg.table_row(parent="data_table"):
            for col_name in data.columns:
                cell_value = data.iloc[row_index][col_name]
                dpg.add_text(str(cell_value))

    # Теперь сохраним наш словарь в user_data самой таблицы,
    # чтобы в колбэке знать, какой столбец какой ID имеет.
    dpg.set_item_user_data("data_table", table_col_map)
    # Показываем окно с данными
    dpg.show_item("data_window")

def _hsv_to_rgb(h, s, v):
    """Конвертирует HSV в RGB."""
    if s == 0.0: return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
    if i == 0: return (255*v, 255*t, 255*p)
    if i == 1: return (255*q, 255*v, 255*p)
    if i == 2: return (255*p, 255*v, 255*t)
    if i == 3: return (255*p, 255*q, 255*v)
    if i == 4: return (255*t, 255*p, 255*v)
    if i == 5: return (255*v, 255*p, 255*q)

def train_model_callback():
    """Обучает модель и выводит результаты."""
    data = dpg.get_item_user_data("target_column_combo") # type: pd.DataFrame
    if data is None:
        add_log("Пожалуйста, загрузите данные.", color=[255, 0, 0])
        return

    target_column = dpg.get_value("target_column_combo")
    if target_column is None or target_column == "":
        add_log("Пожалуйста, выберите целевую переменную.", color=[255, 0, 0])
        return
    test_size = dpg.get_value("test_size_slider") / 100.0
    n_estimators = int(dpg.get_value("n_estimators_input"))
    criterion = dpg.get_value("criterion_combo")
    max_depth = int(dpg.get_value("max_depth_input"))
    min_samples_split = int(dpg.get_value("min_samples_split_input"))
    min_samples_leaf = int(dpg.get_value("min_samples_leaf_input"))
    max_features = dpg.get_value("max_features_combo")
    bootstrap = dpg.get_value("bootstrap_checkbox")
    oob_score = dpg.get_value("oob_score_checkbox")
    n_jobs = int(dpg.get_value("n_jobs_input"))

    results = train_model(
        data,
        target_column,
        test_size,
        n_estimators,
        criterion,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap,
        oob_score,
        n_jobs,
    )

    add_log(f"Точность модели: {results['accuracy']:.2f}", clear=True, color=[0, 255, 0])
    add_log(f"Время обучения: {results['train_time']:.2f} секунд", color=[0, 255, 0], clear=False)
    add_log(f"Размер сохраненной модели: {results['model_size_kb']:.2f} КБ", color=[0, 255, 0], clear=False)

    # Визуализация важности признаков
    feature_importances = results['feature_importances']
    features = pd.get_dummies(data.drop(columns=[target_column])).columns
    fi_df = pd.DataFrame({'Признак': features, 'Важность': feature_importances})
    fi_df = fi_df.sort_values(by='Важность', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Важность', y='Признак', data=fi_df)
    plt.title('Важность признаков')
    plt.tight_layout()
    plt.show()

def add_log(message, color=[0, 0, 0], clear=True):
    if clear:
        dpg.delete_item("log_window", children_only=True)
    dpg.add_text(message, color=color, parent="log_window")
    dpg.set_y_scroll("log_window", dpg.get_y_scroll_max("log_window"))

dpg.create_context()

# Установка шрифта по умолчанию для приложения
with dpg.font_registry():
    with dpg.font(default_font, 18) as default_font:
        dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
        dpg.bind_font(default_font)

with dpg.window(label="Random Forest Classifier", tag="Main Window"):
    
    with dpg.menu_bar(), dpg.menu(label="Menu"):
        dpg.add_text("Меню приложения")
        with dpg.menu(label="Настройки"), dpg.menu(label="Шрифт"):
            dpg.add_menu_item(label="Выбрать шрифт", callback=lambda:dpg.show_tool(dpg.mvTool_Font))
    
    dpg.add_text("Выберите CSV файл с данными для обучения модели:")
    with dpg.group(horizontal=True):
        dpg.add_button(label="Загрузить файл", callback=select_file_for_training_the_model)
        dpg.add_text("", tag="file_name")

    dpg.add_separator()
    
    with dpg.child_window(height=300, horizontal_scrollbar=True, tag="data_window", show=False):
        dpg.add_text("Данные:")
        with dpg.table(
            tag="data_table", header_row=True, no_host_extendX=True, delay_search=True, borders_innerH=True,
            borders_outerH=True, borders_innerV=True, borders_outerV=True, context_menu_in_body=True,
            row_background=True, policy=dpg.mvTable_SizingFixedFit, height=200, scrollY=True, scrollX=True,
            sortable=True, callback=choise_callback, user_data={},
            ):
                pass

        dpg.add_text("Выберите целевую переменную:")
        dpg.add_combo(items=[], tag="target_column_combo")

    dpg.add_separator()

    dpg.add_text("Параметры модели:")
    dpg.add_input_text(label="n_estimators", default_value="100", tag="n_estimators_input")
    with dpg.tooltip("n_estimators_input"):
        dpg.add_text("Количество деревьев в лесу.")
    dpg.add_combo(label="criterion", items=["gini", "entropy"], default_value="gini", tag="criterion_combo")
    with dpg.tooltip("criterion_combo"):
        dpg.add_text("Критерий для оценки качества разбиения.")
    dpg.add_input_text(label="max_depth", default_value="0", tag="max_depth_input")
    with dpg.tooltip("max_depth_input"):
        dpg.add_text("Максимальная глубина дерева. \nЕсли не указано, деревья будут расти до тех пор,\nпока все листья не станут чистыми или\nпока в листе не окажется меньше минимального числа образцов для разделения.")
    dpg.add_input_text(label="min_samples_split", default_value="2", tag="min_samples_split_input")
    with dpg.tooltip("min_samples_split_input"):
        dpg.add_text("Минимальное количество образцов,\nнеобходимое для разделения узла.")
    dpg.add_input_text(label="min_samples_leaf", default_value="1", tag="min_samples_leaf_input")
    with dpg.tooltip("min_samples_leaf_input"):
        dpg.add_text("Минимальное количество образцов,\nкоторое должно быть в листовом узле.")
    dpg.add_combo(label="max_features", items=["None", "sqrt", "log2"], default_value="None", tag="max_features_combo")
    with dpg.tooltip("max_features_combo"):
        dpg.add_text("Максимальное количество признаков,\nрассматриваемых при поиске лучшего разбиения.")
    dpg.add_checkbox(label="bootstrap", default_value=True, tag="bootstrap_checkbox")
    with dpg.tooltip("bootstrap_checkbox"):
        dpg.add_text("Определяет, используется ли бутстреппинг\nпри выборке подвыборок.")
    dpg.add_checkbox(label="oob_score", default_value=False, tag="oob_score_checkbox")
    with dpg.tooltip("oob_score_checkbox"):
        dpg.add_text("Если установлено в True,\nто модель будет использовать out-of-bag\nобразцы для оценки точности.")
    dpg.add_input_text(label="n_jobs", default_value="1", tag="n_jobs_input")
    with dpg.tooltip("n_jobs_input"):
        dpg.add_text("Количество ядер процессора,\nиспользуемых при обучении и предсказании.\nЗначение -1 задействует все доступные ядра.")

    dpg.add_text("Выберите процент данных для тестирования:")
    dpg.add_slider_int(default_value=30, min_value=5, max_value=50, tag="test_size_slider")

    with dpg.theme(tag="green_button"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(2/7.0, 0.6, 0.6))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(2/7.0, 0.8, 0.8))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(2/7.0, 0.7, 0.7))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 2*5)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 2*3, 2*3)

    dpg.add_button(label="Обучить модель", callback=train_model_callback)
    dpg.bind_item_theme(dpg.last_item(), "green_button")

    dpg.add_separator()

    dpg.add_text("Лог:")
    with dpg.child_window(tag="log_window", autosize_x=True, height=150):
        pass

dpg.create_viewport(title='Random Forest Classifier', width=700, height=700)
dpg.setup_dearpygui()

dpg.show_viewport()
dpg.set_primary_window("Main Window", True)
dpg.start_dearpygui()
dpg.destroy_context()
