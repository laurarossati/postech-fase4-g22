# Sistema de Classificação de Obesidade

Projeto Streamlit com:

- predição individual em sete classes;
- upload e predição de arquivos CSV em lote;
- download do CSV com classificações e probabilidades;
- dashboard específico para os dados enviados;
- probabilidades por classe;
- dashboard da base de treinamento com filtros;
- métricas de treino, validação e teste;
- importância global das features.

O modelo final é um XGBoost multiclasse com acurácia de 96,82% no conjunto de
teste holdout.

## Uso simples no Windows

1. Extraia todo o ZIP.
2. Execute `1_INSTALAR.bat`.
3. Execute `2_TESTAR.bat`.
4. Execute `3_INICIAR_APP.bat`.

O navegador abrirá em `http://localhost:8501`.

## Como usar a predição em lote

1. Abra a opção **Predição em lote** no menu lateral ou na página inicial.
2. Envie um CSV com as nove colunas solicitadas.
3. Confira as classificações na aba **Resultados e download**.
4. Baixe o resultado completo, com as probabilidades das sete classes.
5. Abra a aba **Dashboard dos dados enviados** para analisar a nova base.

A própria página disponibiliza um CSV de exemplo. Colunas adicionais são
preservadas no arquivo de resultado.

## Uso pelo terminal

No Windows:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python verificar_projeto.py
python -m streamlit run app.py
```

No macOS ou Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verificar_projeto.py
python -m streamlit run app.py
```

## Estrutura

```text
projeto_streamlit_obesidade/
├── app.py
├── pages/
│   ├── 1_Predicao.py
│   ├── 2_Predicao_em_Lote.py
│   └── 3_Dashboard.py
├── src/
│   ├── config.py
│   ├── model_service.py
│   └── preprocessing.py
├── artifacts/
│   └── modelo_xgboost_obesidade.joblib
├── data/
│   └── Obesity.csv
├── verificar_projeto.py
└── requirements.txt
```

## Como testar

Execute:

```bash
python verificar_projeto.py
```

O teste:

- verifica os arquivos obrigatórios;
- carrega o artefato;
- reproduz a acurácia do teste;
- realiza uma predição de exemplo;
- realiza uma predição em lote;
- confirma a existência das sete probabilidades.

## Publicação

Para publicar:

1. Crie um repositório no GitHub.
2. Envie todo o conteúdo desta pasta para o repositório.
3. No Streamlit Community Cloud, crie uma aplicação selecionando o repositório,
   a branch principal e o arquivo `app.py`.
4. Após o deploy, registre os endereços em `links_entrega.txt`.

## Limitações

- A base é acadêmica e parcialmente sintética.
- O modelo classifica o estado atual; não prevê obesidade futura.
- As associações do dashboard não demonstram causalidade.
- O resultado não substitui avaliação ou diagnóstico profissional.
