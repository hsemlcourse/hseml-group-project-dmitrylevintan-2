import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Банковский скоринг", page_icon="🏦", layout="wide")

st.title("🏦 Система банковского скоринга")
st.caption("Оценка вероятности одобрения кредита по ключевым параметрам клиента.")

MODEL_PATH = "models/loan_scoring_pipeline.pkl"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Не найден файл модели: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


model = load_model()
expected_cols = list(getattr(model, "feature_names_in_", []))

st.subheader("Основные параметры клиента")

c1, c2 = st.columns(2)

with c1:
    age = st.number_input("Возраст", min_value=18, max_value=100, value=30, step=1)
    monthly_income = st.number_input("Доход в месяц", min_value=0.0, value=5000.0, step=100.0)
    loan_amount = st.number_input("Сумма кредита", min_value=0.0, value=10000.0, step=500.0)

with c2:
    dti = st.slider("Коэффициент долговой нагрузки (DTI)", 0.0, 1.0, 0.30, 0.01)
    loan_duration = st.number_input("Срок кредита (мес.)", min_value=1, value=36, step=1)
    credit_history = st.number_input("Кредитная история (лет)", min_value=0.0, value=5.0, step=0.5)

with st.sidebar:
    st.header("Параметры решения")
    threshold = st.slider("Порог одобрения", 0.0, 1.0, 0.50, 0.01)
    st.markdown("---")
    if expected_cols:
        st.write("Признаки модели:")
        st.code(", ".join(expected_cols))

if st.button("Оценить заявку", use_container_width=True):
    try:
        row = {
            "Age": age,
            "MonthlyIncome": monthly_income,
            "LoanAmount": loan_amount,
            "TotalDebtToIncomeRatio": dti,
            "LoanDuration": loan_duration,
            "LengthOfCreditHistory": credit_history,
        }

        df = pd.DataFrame([row])

        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        if expected_cols:
            df = df[expected_cols]

        with st.spinner("Считаю вероятность..."):
            prediction = int(model.predict(df)[0])
            probability = float(model.predict_proba(df)[0][1])

        st.markdown("### Результат")

        a, b, c = st.columns(3)
        a.metric("Вероятность одобрения", f"{probability:.1%}")
        b.metric("Решение", "Одобрено" if prediction == 1 else "Отказ")
        c.metric("Порог", f"{threshold:.0%}")

        st.progress(min(max(probability, 0.0), 1.0))

        if probability >= threshold:
            st.success(f"✅ Кредит одобрен. Вероятность: {probability:.1%}")
        else:
            st.error(f"❌ Кредит не одобрен. Вероятность: {probability:.1%}")

        with st.expander("Посмотреть данные, отправленные в модель"):
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка при расчёте: {e}")