import streamlit as st

from src.model_service import load_artifact


st.set_page_config(
    page_title="Classificação de Obesidade",
    page_icon="🩺",
    layout="wide",
)

artifact = load_artifact()
test_metrics = artifact["metricas_treinamento"].loc["Teste"]

st.title("Sistema de Apoio à Classificação de Obesidade")
st.write(
    "Aplicação acadêmica que utiliza um modelo XGBoost multiclasse para "
    "estimar o nível de obesidade a partir de informações físicas e "
    "comportamentais."
)

col1, col2, col3 = st.columns(3)
col1.metric("Acurácia no teste", f"{test_metrics['Accuracy']:.1%}")
col2.metric("Classes previstas", "7")
col3.metric("Features do modelo", len(artifact["colunas_input"]))

st.divider()
st.subheader("Escolha uma opção")

individual_col, batch_col, dashboard_col = st.columns(3)
with individual_col:
    st.markdown("### 🧍 Predição individual")
    st.write(
        "Informe os dados de uma pessoa e veja a classe estimada e as "
        "probabilidades das sete classes."
    )
    st.page_link(
        "pages/1_Predicao.py",
        label="Abrir predição",
        icon="➡️",
    )

with batch_col:
    st.markdown("### 📄 Predição em lote")
    st.write(
        "Envie um CSV, classifique todos os registros, baixe o resultado e "
        "analise os novos dados."
    )
    st.page_link(
        "pages/2_Predicao_em_Lote.py",
        label="Abrir predição em lote",
        icon="➡️",
    )

with dashboard_col:
    st.markdown("### 📊 Dashboard analítico")
    st.write(
        "Explore a base de treinamento, os principais indicadores e o "
        "desempenho do modelo."
    )
    st.page_link(
        "pages/3_Dashboard.py",
        label="Abrir dashboard",
        icon="➡️",
    )

st.divider()
st.info(
    "Este sistema foi desenvolvido para fins acadêmicos e de apoio analítico. "
    "O resultado não substitui avaliação ou diagnóstico realizado por "
    "profissionais de saúde."
)
