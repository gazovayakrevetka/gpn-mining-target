## Проект: Прогноз Target в данных горной добычи

Этот проект реализует полный цикл:

- **EDA** (разведочный анализ) – `1_eda.py`
- **Предобработка и разбиение** – `2_preprocessing.py`
- **Обучение и выбор лучшей модели** – `3_model_training.py`
- **Streamlit‑приложение** – `app.py`

Все файлы находятся в папке `GPN2`.

### Структура данных

Ожидается таблица с колонками:

- `Block_ID`
- `X`, `Y`, `Z`
- `Rock_Type`
- `Ore_Grade (%)`
- `Tonnage`
- `Ore_Value (USD/tonne)`
- `Mining_Cost (USD)`
- `Processing_Cost (USD)`
- `Waste_Flag`
- `Profit (USD)`
- `Target` (целевой столбец, регрессия)

Файл с данными:

- `GPN2/mining_block_model.xlsx` (рекомендуется)  
  или  
- `GPN2/mining_block_model.csv`

---

## Установка и локальный запуск

### 1. Подготовка окружения

```bash
cd GPN2

python -m venv .venv
.venv\Scripts\activate  # Windows
# или
source .venv/bin/activate  # Linux / macOS

pip install -r requirements.txt
```

Положите файл `mining_block_model.xlsx` (или `mining_block_model.csv`) в папку `GPN2`.

---

### 2. Последовательность запуска скриптов

1. **EDA**

   ```bash
   python 1_eda.py
   ```

   В папке `images/` появятся гистограммы, boxplot'ы, корреляционная матрица,
   графики связи `Rock_Type` с `Target`.

2. **Предобработка**

   ```bash
   python 2_preprocessing.py
   ```

   Скрипт:

   - очищает данные и создаёт признак `Total_Cost_per_tonne`;
   - делит на train/test (80/20, `random_state=42`);
   - масштабирует числовые признаки (`StandardScaler`);
   - кодирует `Rock_Type` (`OneHotEncoder`);
   - сохраняет:
     - `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`;
     - `encoder.pkl`, `scaler.pkl`;
     - `preprocessing_metadata.pkl`.

3. **Обучение моделей**

   ```bash
   python 3_model_training.py
   ```

   Скрипт:

   - обучает модели:
     - `LinearRegression`
     - `RandomForestRegressor`
     - `XGBRegressor`
     - `MLPRegressor`
   - оценивает их по 5‑fold CV (RMSE, R²);
   - подбирает гиперпараметры для двух лучших по RMSE;
   - сравнивает на тестовой выборке (RMSE, MAE, R²);
   - сохраняет:
     - `best_model.pkl`;
     - `model_performance.txt`, `model_performance.json`.

---

### 3. Запуск Streamlit‑приложения локально

```bash
cd GPN2
streamlit run app.py
```

Приложение позволяет:

- вводить признаки одного блока и получать прогноз **Target**;
- загружать CSV/XLSX с несколькими блоками и скачивать результаты с колонкой `Predicted_Target`;
- при наличии `Target` в файле строить график «предсказание против факта»;
- просматривать метрики модели и описание пайплайна.

---

## Деплой на Streamlit Cloud

1. Инициализируйте git‑репозиторий в корне (при необходимости перенесите содержимое `GPN2` в корень будущего репозитория).

   ```bash
   git init
   git add .
   git commit -m "Initial commit: mining Target prediction"
   git remote add origin https://github.com/<логин>/<репозиторий>.git
   git push -u origin main
   ```

2. На [Streamlit Cloud](https://streamlit.io/cloud):

   - создайте новое приложение (**New app**);
   - выберите репозиторий и ветку (`main`);
   - укажите `app.py` в качестве основного файла (если файлы лежат в `GPN2`, укажите `GPN2/app.py`);
   - убедитесь, что `requirements.txt` доступен (в корне проекта или в `GPN2`).

3. Нажмите **Deploy**.

После деплоя вставьте ссылку на приложение сюда:

```markdown
### Демонстрационное приложение

Готовое приложение на Streamlit Cloud: <ссылка на ваше приложение>
```

---

## Файлы проекта

- `1_eda.py` – разведочный анализ, сохранение графиков в `images/`.
- `2_preprocessing.py` – предобработка, сохранение `encoder.pkl`, `scaler.pkl` и train/test.
- `3_model_training.py` – обучение нескольких моделей, выбор лучшей, сохранение `best_model.pkl` и метрик.
- `app.py` – Streamlit‑приложение для прогноза Target.
- `requirements.txt` – зависимости.
- `.gitignore` – игнорируемые файлы.
- `ASSUMPTIONS_AND_DECISIONS.md` – допущения и принятые решения.

