[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — [Предсказание одобрения кредита]
P.S в ведомости проект называется [Предсказание одобрения кредитной земли] Видимо кто-то решил пошутить или случайно заполнил.

**Студент:** Левинтан Дмитрий Маркович

**Группа:**  БИВ 234


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

В данном проекте решается задача бинарной классификации банковского скоринга: на основе социально-демографических характеристик заемщика и его текущих финансовых показателей необходимо предсказать, будет ли одобрена кредитная заявка (`LoanApproved`).


---
**Задача:** Бинарная классификация (одобрено `1` / отказано `0`).

**Датасет:** Financial Risk for Loan Approval (https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?resource=download&select=Loan.csv)

**Целевая метрика:** `F1-score` (основная для оптимизации в условиях дисбаланса классов) и `ROC-AUC` (для оценки качества ранжирования заемщиков).


## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda_and_modeling.ipynb            # выполнение первой части
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
├── tests
├── requirements.txt
└── README.md
```

## Запуск

Этот блок замените способом запуска вашего сервиса.
```bash
# 1. Клонирование репозитория
git clone [https://github.com/hsemlcourse/hseml-group-project-dmitrylevintan-2.git](https://github.com/hsemlcourse/hseml-group-project-dmitrylevintan-2.git)
cd hseml-group-project-dmitrylevintan-2

# 2. Настройка виртуального окружения
python -bin venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate

# 3. Установка необходимых зависимостей
pip install -r requirements.txt

# 4. Запуск ноутбука с экспериментами
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```



## Результаты
| Модель | Best Hyperparameters | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `{'solver': 'lbfgs', 'C': 100}` | **0.9332** | **0.8525** | **0.8143** | **0.8330** | **0.9791** |
| **XGBoost** | `{'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1}` | 0.9277 | 0.8483 | 0.7875 | 0.8167 | 0.9750 |
| **Gradient Boosting** | `{'subsample': 0.8, 'n_estimators': 150, 'max_depth': 4}` | 0.9274 | 0.8500 | 0.7836 | 0.8155 | 0.9741 |
| **LightGBM** | `{'num_leaves': 31, 'n_estimators': 100, 'learning_rate': 0.05}` | 0.9256 | 0.8476 | 0.7759 | 0.8102 | 0.9752 |
| **Random Forest** | `{'n_estimators': 300, 'min_samples_split': 5, 'max_depth': 10}` | 0.9080 | 0.8370 | 0.6837 | 0.7526 | 0.9588 |
| **KNN** | `{'weights': 'distance', 'n_neighbors': 11, 'metric': 'manhattan'}` | 0.8779 | 0.8000 | 0.5378 | 0.6432 | 0.9041 |

##Этапы реализации исследования
Устранение утечки данных (Data Leakage): Из датасета полностью исключены признаки InterestRate, BaseInterestRate и MonthlyLoanPayment. Данные фичи формируются после вынесения банком решения об одобрении и приводили к искусственному завышению метрик.

Борьба с мультиколлинеарностью: Удален признак годового дохода (AnnualIncome), так как он линейно дублировал месячный доход (MonthlyIncome). Это позволило сделать оценки важности признаков математически объективными.

Очистка и Feature Engineering: Выбросы в доходах обработаны с помощью метода межквартильного размаха (IQR). Сгенерированы новые экономически обоснованные признаки: IncomePerDependent (удельный доход на члена семьи), CreditHistoryToAge (зрелость кредитной истории) и AssetsToLiabilities (коэффициент покрытия обязательств активами).

Архитектура препроцессинга: Все трансформации (заполнение пропусков, масштабирование StandardScaler и кодирование OneHotEncoder) инкапсулированы в изолированный ColumnTransformer внутри Sklearn-пайплайнов, что исключает переобучение при кросс-валидации.
## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
