import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import CLASS_COLORS
from src.model_service import load_artifact, predict_patient


st.set_page_config(
    page_title="Predição individual",
    page_icon="🧍",
    layout="wide",
)

st.title("Predição individual")
st.write(
    "Preencha os campos abaixo. O modelo calculará as probabilidades para "
    "cada uma das sete classes."
)

gender_options = {
    "Feminino": "Female",
    "Masculino": "Male",
}
yes_no_options = {
    "Sim": "yes",
    "Não": "no",
}
frequency_options = {
    "Nunca": "no",
    "Às vezes": "Sometimes",
    "Frequentemente": "Frequently",
    "Sempre": "Always",
}
transport_options = {
    "Transporte público": "Public_Transportation",
    "Automóvel": "Automobile",
    "Caminhada": "Walking",
    "Bicicleta": "Bike",
    "Motocicleta": "Motorbike",
}

with st.form("patient_form"):
    left, right = st.columns(2)

    with left:
        age = st.number_input(
            "Idade (anos)",
            min_value=14,
            max_value=100,
            value=23,
            step=1,
        )
        height = st.number_input(
            "Altura (metros)",
            min_value=1.20,
            max_value=2.30,
            value=1.70,
            step=0.01,
            format="%.2f",
        )
        weight = st.number_input(
            "Peso (kg)",
            min_value=30.0,
            max_value=250.0,
            value=83.0,
            step=0.5,
            format="%.1f",
        )
        gender_label = st.selectbox(
            "Gênero informado na base",
            options=list(gender_options),
        )
        family_history_label = st.selectbox(
            "Há histórico familiar de excesso de peso?",
            options=list(yes_no_options),
        )

    with right:
        fcvc = st.selectbox(
            "Frequência de consumo de vegetais",
            options=[1, 2, 3],
            index=1,
            format_func=lambda value: {
                1: "Baixa",
                2: "Média",
                3: "Alta",
            }[value],
        )
        caec_label = st.selectbox(
            "Consumo de alimentos entre refeições",
            options=list(frequency_options),
            index=1,
        )
        calc_label = st.selectbox(
            "Consumo de bebida alcoólica",
            options=list(frequency_options),
            index=1,
        )
        transport_label = st.selectbox(
            "Meio de transporte mais utilizado",
            options=list(transport_options),
        )

    submitted = st.form_submit_button(
        "Calcular classificação",
        type="primary",
        width="stretch",
    )

if submitted:
    raw_patient = pd.DataFrame(
        [
            {
                "Age": age,
                "Height": height,
                "Weight": weight,
                "Gender": gender_options[gender_label],
                "family_history": yes_no_options[family_history_label],
                "FCVC": fcvc,
                "CAEC": frequency_options[caec_label],
                "CALC": frequency_options[calc_label],
                "MTRANS": transport_options[transport_label],
            }
        ]
    )

    try:
        result = predict_patient(raw_patient)
    except (ValueError, FileNotFoundError) as error:
        st.error(f"Não foi possível realizar a predição: {error}")
        st.stop()

    bmi = weight / (height**2)
    probability_pct = result["probabilidade"] * 100

    st.divider()
    st.subheader("Resultado")

    result_col, probability_col, bmi_col = st.columns(3)
    result_col.metric("Classe estimada", result["classe"])
    probability_col.metric(
        "Probabilidade da classe",
        f"{probability_pct:.1f}%",
    )
    bmi_col.metric("BMI calculado", f"{bmi:.1f} kg/m²")

    outside_training_range = (
        age > 61
        or height < 1.45
        or height > 1.98
        or weight < 39
        or weight > 173
    )
    if outside_training_range:
        st.warning(
            "Um ou mais valores estão fora da faixa observada na base de "
            "treinamento. A estimativa deve ser interpretada com cautela."
        )

    probabilities = result["probabilidades"].sort_values(
        "probabilidade_pct",
        ascending=True,
    )
    figure = px.bar(
        probabilities,
        x="probabilidade_pct",
        y="classe",
        orientation="h",
        color="classe",
        color_discrete_map=CLASS_COLORS,
        text="probabilidade_pct",
        labels={
            "probabilidade_pct": "Probabilidade (%)",
            "classe": "Classe",
        },
        title="Probabilidade estimada para cada classe",
    )
    figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )
    figure.update_layout(
        showlegend=False,
        xaxis_range=[0, max(100, probabilities["probabilidade_pct"].max() * 1.1)],
        yaxis_title="",
    )
    st.plotly_chart(figure, width="stretch")

    with st.expander("Ver as oito features enviadas ao modelo"):
        st.dataframe(
            result["model_input"],
            width="stretch",
            hide_index=True,
        )

st.info(
    "Resultado acadêmico de apoio analítico. Não utilize esta classificação "
    "como diagnóstico ou recomendação médica."
)

with st.expander("Informações sobre o modelo"):
    artifact = load_artifact()
    st.write(
        "Modelo XGBoost multiclasse treinado com oito features. "
        f"Acurácia no teste holdout: "
        f"{artifact['metricas_treinamento'].loc['Teste', 'Accuracy']:.1%}."
    )
