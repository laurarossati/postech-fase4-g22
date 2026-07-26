from __future__ import annotations

import csv
import hashlib
from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
    DATA_PATH,
    OBESITY_CLASSES,
    RAW_REQUIRED_COLUMNS,
)
from src.model_service import predict_batch


st.set_page_config(
    page_title="Predição em lote",
    page_icon="📄",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Lê CSV com vírgula, ponto e vírgula ou tabulação."""
    last_error: Exception | None = None

    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            text = file_bytes.decode(encoding)
            try:
                separator = csv.Sniffer().sniff(
                    text[:10_000],
                    delimiters=",;\t",
                ).delimiter
            except csv.Error:
                separator = ","

            data = pd.read_csv(StringIO(text), sep=separator)
            if len(data.columns) == 1 and separator != ";":
                data = pd.read_csv(StringIO(text), sep=";")
            return data
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            last_error = error

    raise ValueError(
        "Não foi possível ler o CSV. Salve-o como CSV UTF-8."
    ) from last_error


@st.cache_data(show_spinner=False)
def example_csv() -> bytes:
    example = pd.read_csv(DATA_PATH)[RAW_REQUIRED_COLUMNS].head(5)
    return example.to_csv(index=False).encode("utf-8-sig")


st.title("Predição em lote")
st.write(
    "Envie um arquivo CSV para classificar várias pessoas de uma só vez. "
    "Após o cálculo, você poderá baixar os resultados e explorar um "
    "dashboard criado com os dados enviados."
)

with st.expander("Formato esperado do arquivo"):
    st.write("O CSV precisa conter estas nove colunas:")
    st.code(", ".join(RAW_REQUIRED_COLUMNS), language=None)
    st.write(
        "Colunas adicionais, como `Obesity`, podem permanecer no arquivo e "
        "serão preservadas no resultado."
    )
    st.download_button(
        "Baixar CSV de exemplo",
        data=example_csv(),
        file_name="exemplo_predicao_lote.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader(
    "Selecione o arquivo CSV",
    type=["csv"],
    help="São aceitos arquivos separados por vírgula ou ponto e vírgula.",
)

if uploaded_file is None:
    st.info(
        "Baixe o exemplo acima ou envie o arquivo Obesity.csv para testar."
    )
    st.stop()

file_bytes = uploaded_file.getvalue()
file_key = hashlib.sha256(file_bytes).hexdigest()[:10]

try:
    raw_data = read_csv_bytes(file_bytes)
except (ValueError, pd.errors.EmptyDataError) as error:
    st.error(f"Não foi possível abrir o arquivo: {error}")
    st.stop()

if raw_data.empty:
    st.error("O arquivo CSV está vazio.")
    st.stop()

with st.expander("Conferir dados recebidos"):
    record_count = f"{len(raw_data):,}".replace(",", ".")
    st.write(
        f"**{record_count} registros** e "
        f"**{len(raw_data.columns)} colunas**."
    )
    st.dataframe(raw_data.head(20), width="stretch", hide_index=True)

try:
    with st.spinner("Calculando as classificações..."):
        batch = predict_batch(raw_data)
except (ValueError, FileNotFoundError) as error:
    st.error(f"Não foi possível realizar a predição: {error}")
    st.stop()

results = batch["resultados"]
model_input = batch["model_input"]
record_count = f"{len(results):,}".replace(",", ".")

st.success(f"{record_count} registros classificados com sucesso.")

results_tab, dashboard_tab = st.tabs(
    ["Resultados e download", "Dashboard dos dados enviados"]
)

with results_tab:
    st.subheader("Resultado das classificações")

    result_columns = [
        "linha_csv",
        "Age",
        "Height",
        "Weight",
        "BMI_calculado",
        "classe_predita",
        "probabilidade_predita_pct",
    ]
    visible_columns = [
        column for column in result_columns if column in results.columns
    ]

    st.dataframe(
        results[visible_columns].style.format(
            {
                "BMI_calculado": "{:.2f}",
                "probabilidade_predita_pct": "{:.1f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "O arquivo completo também contém as probabilidades das sete classes."
    )
    output_csv = results.to_csv(
        index=False,
        sep=";",
        decimal=",",
    ).encode("utf-8-sig")
    st.download_button(
        "Baixar resultado completo",
        data=output_csv,
        file_name="predicoes_obesidade_lote.csv",
        mime="text/csv",
        type="primary",
    )

with dashboard_tab:
    dashboard_data = results.copy()
    dashboard_data["Age_modelo"] = model_input["Age"].to_numpy()
    dashboard_data["genero"] = (
        dashboard_data["Gender"]
        .astype("string")
        .str.strip()
        .map({"Female": "Feminino", "Male": "Masculino"})
    )
    dashboard_data["historico_familiar"] = (
        dashboard_data["family_history"]
        .astype("string")
        .str.strip()
        .map({"yes": "Sim", "no": "Não"})
    )
    class_labels_ordered = [
        CLASS_LABELS[name] for name in CLASS_ORDER
    ]
    dashboard_data["classe_predita"] = pd.Categorical(
        dashboard_data["classe_predita"],
        categories=class_labels_ordered,
        ordered=True,
    )
    dashboard_data["obesidade_predita"] = dashboard_data[
        "classe_predita_original"
    ].isin(OBESITY_CLASSES)

    st.subheader("Filtros")
    filter1, filter2, filter3 = st.columns(3)

    present_classes = set(
        dashboard_data["classe_predita"].dropna().astype(str).unique()
    )
    available_classes = [
        class_name
        for class_name in class_labels_ordered
        if class_name in present_classes
    ]
    available_genders = sorted(
        dashboard_data["genero"].dropna().unique().tolist()
    )
    available_history = sorted(
        dashboard_data["historico_familiar"].dropna().unique().tolist()
    )

    with filter1:
        selected_classes = st.multiselect(
            "Classe prevista",
            options=available_classes,
            default=available_classes,
            key=f"batch_classes_{file_key}",
        )
    with filter2:
        selected_genders = st.multiselect(
            "Gênero informado",
            options=available_genders,
            default=available_genders,
            key=f"batch_genders_{file_key}",
        )
    with filter3:
        selected_history = st.multiselect(
            "Histórico familiar",
            options=available_history,
            default=available_history,
            key=f"batch_history_{file_key}",
        )

    filtered = dashboard_data[
        dashboard_data["classe_predita"].isin(selected_classes)
        & dashboard_data["genero"].isin(selected_genders)
        & dashboard_data["historico_familiar"].isin(selected_history)
    ].copy()

    if filtered.empty:
        st.warning("Nenhum registro corresponde aos filtros selecionados.")
        st.stop()

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric(
        "Registros",
        f"{len(filtered):,}".replace(",", "."),
    )
    kpi2.metric(
        "Idade média",
        f"{filtered['Age_modelo'].mean():.1f} anos",
    )
    kpi3.metric(
        "BMI mediano",
        f"{filtered['BMI_calculado'].median():.1f} kg/m²",
    )
    kpi4.metric(
        "Classes de obesidade I-III",
        f"{filtered['obesidade_predita'].mean():.1%}",
    )
    kpi5.metric(
        "Confiança média",
        f"{filtered['probabilidade_predita_pct'].mean():.1f}%",
    )

    distribution = (
        filtered.groupby("classe_predita", observed=False)
        .size()
        .reindex(class_labels_ordered, fill_value=0)
        .rename("quantidade")
        .reset_index()
    )
    distribution["percentual"] = (
        distribution["quantidade"]
        / distribution["quantidade"].sum()
        * 100
    )
    distribution_chart = px.bar(
        distribution,
        x="classe_predita",
        y="quantidade",
        color="classe_predita",
        color_discrete_map=CLASS_COLORS,
        text="percentual",
        labels={
            "classe_predita": "Classe prevista",
            "quantidade": "Quantidade",
        },
        title="Distribuição das classes previstas",
    )
    distribution_chart.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )
    distribution_chart.update_layout(
        showlegend=False,
        xaxis_tickangle=-25,
    )
    st.plotly_chart(distribution_chart, width="stretch")

    left, right = st.columns(2)

    with left:
        bmi_chart = px.box(
            filtered,
            x="classe_predita",
            y="BMI_calculado",
            color="classe_predita",
            color_discrete_map=CLASS_COLORS,
            category_orders={
                "classe_predita": class_labels_ordered,
            },
            labels={
                "classe_predita": "Classe prevista",
                "BMI_calculado": "BMI (kg/m²)",
            },
            title="Distribuição de BMI por classe prevista",
        )
        bmi_chart.update_layout(
            showlegend=False,
            xaxis_tickangle=-30,
        )
        st.plotly_chart(bmi_chart, width="stretch")

    with right:
        age_bmi_chart = px.scatter(
            filtered,
            x="Age_modelo",
            y="BMI_calculado",
            color="classe_predita",
            color_discrete_map=CLASS_COLORS,
            category_orders={
                "classe_predita": class_labels_ordered,
            },
            opacity=0.65,
            labels={
                "Age_modelo": "Idade",
                "BMI_calculado": "BMI (kg/m²)",
                "classe_predita": "Classe prevista",
            },
            title="Relação entre idade, BMI e classe prevista",
        )
        st.plotly_chart(age_bmi_chart, width="stretch")

    left, right = st.columns(2)

    with left:
        gender_distribution = (
            filtered.groupby(
                ["genero", "classe_predita"],
                observed=True,
            )
            .size()
            .rename("quantidade")
            .reset_index()
        )
        gender_chart = px.bar(
            gender_distribution,
            x="genero",
            y="quantidade",
            color="classe_predita",
            color_discrete_map=CLASS_COLORS,
            category_orders={
                "classe_predita": class_labels_ordered,
            },
            labels={
                "genero": "Gênero informado",
                "quantidade": "Quantidade",
                "classe_predita": "Classe prevista",
            },
            title="Classes previstas por gênero informado",
            barmode="stack",
        )
        st.plotly_chart(gender_chart, width="stretch")

    with right:
        family_summary = (
            filtered.groupby("historico_familiar")[
                "obesidade_predita"
            ]
            .mean()
            .mul(100)
            .rename("percentual_obesidade")
            .reset_index()
        )
        family_chart = px.bar(
            family_summary,
            x="historico_familiar",
            y="percentual_obesidade",
            color="historico_familiar",
            text="percentual_obesidade",
            labels={
                "historico_familiar": "Histórico familiar",
                "percentual_obesidade": (
                    "Registros previstos nas classes de obesidade (%)"
                ),
            },
            title="Classes de obesidade e histórico familiar",
        )
        family_chart.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )
        family_chart.update_layout(
            showlegend=False,
            yaxis_range=[0, 100],
        )
        st.plotly_chart(family_chart, width="stretch")

    st.caption(
        "Os indicadores representam as classificações produzidas pelo modelo "
        "para o arquivo enviado; não representam diagnóstico nem prevalência "
        "da população."
    )
