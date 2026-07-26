import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import classification_report

from src.config import (
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
    DATA_PATH,
    OBESITY_CLASSES,
)
from src.model_service import load_artifact


st.set_page_config(
    page_title="Dashboard analítico",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    for column in data.select_dtypes(include=["object", "string"]).columns:
        data[column] = data[column].astype("string").str.strip()
    data = data.drop_duplicates().copy()
    data["BMI"] = data["Weight"] / (data["Height"] ** 2)
    data["classe"] = data["Obesity"].map(CLASS_LABELS)
    data["classe"] = pd.Categorical(
        data["classe"],
        categories=[CLASS_LABELS[name] for name in CLASS_ORDER],
        ordered=True,
    )
    data["obesidade"] = data["Obesity"].isin(OBESITY_CLASSES)
    data["genero"] = data["Gender"].map(
        {"Female": "Feminino", "Male": "Masculino"}
    )
    data["historico_familiar"] = data["family_history"].map(
        {"yes": "Sim", "no": "Não"}
    )
    return data


data = load_dashboard_data()
artifact = load_artifact()

st.title("Dashboard da base de treinamento")
st.write(
    "Visão exploratória da base utilizada no projeto e dos resultados do "
    "modelo de classificação."
)

st.sidebar.header("Filtros")
selected_genders = st.sidebar.multiselect(
    "Gênero informado na base",
    options=["Feminino", "Masculino"],
    default=["Feminino", "Masculino"],
)
selected_history = st.sidebar.multiselect(
    "Histórico familiar de excesso de peso",
    options=["Sim", "Não"],
    default=["Sim", "Não"],
)
age_range = st.sidebar.slider(
    "Faixa de idade",
    min_value=int(data["Age"].min()),
    max_value=int(np.ceil(data["Age"].max())),
    value=(
        int(data["Age"].min()),
        int(np.ceil(data["Age"].max())),
    ),
)

filtered = data[
    data["genero"].isin(selected_genders)
    & data["historico_familiar"].isin(selected_history)
    & data["Age"].between(age_range[0], age_range[1])
].copy()

if filtered.empty:
    st.warning("Nenhum registro corresponde aos filtros selecionados.")
    st.stop()

obesity_rate = filtered["obesidade"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Registros filtrados", f"{len(filtered):,}".replace(",", "."))
kpi2.metric("Idade média", f"{filtered['Age'].mean():.1f} anos")
kpi3.metric("BMI mediano", f"{filtered['BMI'].median():.1f} kg/m²")
kpi4.metric("Classes de obesidade I-III", f"{obesity_rate:.1%}")

st.caption(
    "A base é acadêmica e parcialmente sintética. As proporções não devem "
    "ser interpretadas como prevalência da população."
)

st.divider()
st.subheader("Perfil da base")

distribution = (
    filtered.groupby("classe", observed=False)
    .size()
    .reindex([CLASS_LABELS[name] for name in CLASS_ORDER], fill_value=0)
    .rename("quantidade")
    .reset_index()
)
distribution["percentual"] = (
    distribution["quantidade"] / distribution["quantidade"].sum() * 100
)

distribution_chart = px.bar(
    distribution,
    x="classe",
    y="quantidade",
    color="classe",
    color_discrete_map=CLASS_COLORS,
    text="percentual",
    labels={"classe": "Classe", "quantidade": "Quantidade"},
    title="Distribuição das classes",
)
distribution_chart.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
)
distribution_chart.update_layout(showlegend=False, xaxis_tickangle=-25)
st.plotly_chart(distribution_chart, width="stretch")

left, right = st.columns(2)

with left:
    bmi_chart = px.box(
        filtered,
        x="classe",
        y="BMI",
        color="classe",
        color_discrete_map=CLASS_COLORS,
        category_orders={
            "classe": [CLASS_LABELS[name] for name in CLASS_ORDER]
        },
        labels={"classe": "Classe", "BMI": "BMI (kg/m²)"},
        title="Distribuição de BMI por classe",
    )
    bmi_chart.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(bmi_chart, width="stretch")

with right:
    activity = (
        filtered.groupby("classe", observed=False)["FAF"]
        .mean()
        .reindex([CLASS_LABELS[name] for name in CLASS_ORDER])
        .rename("atividade_media")
        .reset_index()
    )
    activity_chart = px.bar(
        activity,
        x="classe",
        y="atividade_media",
        color="classe",
        color_discrete_map=CLASS_COLORS,
        labels={
            "classe": "Classe",
            "atividade_media": "Frequência média de atividade física",
        },
        title="Atividade física média por classe",
    )
    activity_chart.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(activity_chart, width="stretch")

left, right = st.columns(2)

with left:
    family_history = (
        filtered.groupby("historico_familiar")["obesidade"]
        .mean()
        .mul(100)
        .rename("percentual_obesidade")
        .reset_index()
    )
    history_chart = px.bar(
        family_history,
        x="historico_familiar",
        y="percentual_obesidade",
        color="historico_familiar",
        text="percentual_obesidade",
        labels={
            "historico_familiar": "Histórico familiar",
            "percentual_obesidade": "Registros em classes de obesidade (%)",
        },
        title="Classes de obesidade e histórico familiar",
    )
    history_chart.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )
    history_chart.update_layout(showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(history_chart, width="stretch")

with right:
    age_bmi_chart = px.scatter(
        filtered,
        x="Age",
        y="BMI",
        color="classe",
        color_discrete_map=CLASS_COLORS,
        category_orders={
            "classe": [CLASS_LABELS[name] for name in CLASS_ORDER]
        },
        opacity=0.65,
        labels={
            "Age": "Idade",
            "BMI": "BMI (kg/m²)",
            "classe": "Classe",
        },
        title="Relação entre idade, BMI e classe",
    )
    st.plotly_chart(age_bmi_chart, width="stretch")

st.divider()
st.subheader("Desempenho do modelo")

metrics = artifact["metricas_treinamento"].copy()
metrics.columns = [
    "Acurácia",
    "Precisão macro",
    "F1 macro",
    "Sensibilidade macro",
    "ROC AUC OVR macro",
]
st.dataframe(
    metrics.style.format("{:.1%}"),
    width="stretch",
)

test_x = artifact["dados"]["X_test"]
test_y = artifact["dados"]["y_test"].squeeze()
test_prediction = artifact["modelo"].predict(test_x)
inverse_target = {
    code: name for name, code in artifact["target_map"].items()
}
model_classes = artifact["modelo"].classes_

report = classification_report(
    test_y,
    test_prediction,
    labels=model_classes,
    target_names=[
        CLASS_LABELS[inverse_target[int(code)]] for code in model_classes
    ],
    output_dict=True,
    zero_division=0,
)
class_report = (
    pd.DataFrame(report)
    .T
    .loc[
        [CLASS_LABELS[inverse_target[int(code)]] for code in model_classes],
        ["precision", "recall", "f1-score", "support"],
    ]
)
class_report.columns = ["Precisão", "Recall", "F1-score", "Registros"]

left, right = st.columns(2)

with left:
    st.markdown("#### Métricas por classe no teste")
    st.dataframe(
        class_report.style.format(
            {
                "Precisão": "{:.1%}",
                "Recall": "{:.1%}",
                "F1-score": "{:.1%}",
                "Registros": "{:.0f}",
            }
        ),
        width="stretch",
    )

with right:
    feature_importance = artifact["feature_importance"].copy()
    feature_names = {
        "BMI": "BMI",
        "flag_female": "Gênero informado",
        "FCVC": "Consumo de vegetais",
        "flag_family_history": "Histórico familiar",
        "CAEC_freq": "Alimentos entre refeições",
        "CALC_freq": "Consumo de álcool",
        "MTRANS_Public_Transportation": "Transporte público",
        "Age": "Idade",
    }
    feature_importance["feature_label"] = feature_importance["feature"].map(
        feature_names
    )
    importance_chart = px.bar(
        feature_importance.sort_values("importance"),
        x="importance",
        y="feature_label",
        orientation="h",
        text="importance",
        labels={
            "importance": "Importância",
            "feature_label": "Feature",
        },
        title="Importância global das features",
    )
    importance_chart.update_traces(texttemplate="%{text:.1%}")
    importance_chart.update_layout(yaxis_title="")
    st.plotly_chart(importance_chart, width="stretch")

st.divider()
st.subheader("Principais leituras para o negócio")

lowest_recall_class = class_report["Recall"].idxmin()
lowest_recall = class_report.loc[lowest_recall_class, "Recall"]
top_feature = artifact["feature_importance"].iloc[0]

st.markdown(
    f"""
- O modelo supera o corte de 75%, com **{metrics.loc['Teste', 'Acurácia']:.1%}
  de acurácia no teste holdout**.
- A feature de maior importância é **{feature_names[top_feature['feature']]}**,
  responsável por aproximadamente **{top_feature['importance']:.1%}** da
  importância global.
- A classe com menor recall no teste é **{lowest_recall_class}**, ainda com
  **{lowest_recall:.1%}**.
- Os padrões apresentados indicam associação dentro desta amostra e não
  demonstram causalidade.
"""
)

download_data = filtered.drop(
    columns=["obesidade"],
    errors="ignore",
).to_csv(index=False).encode("utf-8")
st.download_button(
    "Baixar dados filtrados",
    data=download_data,
    file_name="dados_dashboard_filtrados.csv",
    mime="text/csv",
)
