# Sistema de Classificação de Obesidade

Projeto desenvolvido para o **Tech Challenge da Fase 4 – Data Analytics da FIAP/POSTECH**, pelo Grupo 22.

O projeto utiliza Machine Learning para classificar o nível de obesidade a partir de características físicas, comportamentais e de histórico familiar.

## Aplicação publicada

- **Aplicação Streamlit:** `[Aplicação](https://postech-fase4-g22-nivel-obesidade.streamlit.app/)`
- **Dashboard analítico:** `[Dashboard](https://postech-fase4-g22-nivel-obesidade.streamlit.app/Dashboard)`
- **Repositório GitHub:** [postech-fase4-g22](https://github.com/laurarossati/postech-fase4-g22)
- **Vídeo de apresentação:** `preencher após a gravação`

## Objetivo

Desenvolver uma pipeline completa de Machine Learning e disponibilizar um sistema preditivo capaz de apoiar a análise do nível de obesidade.

A solução contempla:

- análise exploratória;
- preparação e tratamento dos dados;
- engenharia de atributos;
- treinamento e avaliação de modelos;
- seleção do modelo final;
- aplicação preditiva em Streamlit;
- predição individual;
- predição de arquivos CSV em lote;
- dashboard analítico da base de treinamento;
- dashboard dos dados enviados pelo usuário.

## Base de dados

A base original está disponível em:

```text
data/Obesity.csv
```

Ela contém 2.111 registros e 17 variáveis, incluindo informações como:

- gênero;
- idade;
- altura;
- peso;
- histórico familiar;
- hábitos alimentares;
- frequência de atividade física;
- consumo de água e álcool;
- meio de transporte;
- nível de obesidade.

A variável-alvo é `Obesity`, composta por sete classes:

1. Peso insuficiente;
2. Peso normal;
3. Sobrepeso nível I;
4. Sobrepeso nível II;
5. Obesidade tipo I;
6. Obesidade tipo II;
7. Obesidade tipo III.

A base é acadêmica e parcialmente sintética.

## Pipeline de modelagem

Os notebooks estão disponíveis na pasta `modelagem/` e devem ser consultados na seguinte ordem:

| Ordem | Notebook | Descrição |
|---|---|---|
| 1 | `1_exploracao.ipynb` | Análise exploratória e entendimento da base |
| 2 | `2_preparacao.ipynb` | Tratamento, engenharia de atributos e separação dos dados |
| 3 | `3_modelagem.ipynb` | Treinamento, validação e comparação dos modelos |
| 4 | `4_modelo_final.ipynb` | Treinamento, avaliação e persistência do modelo final |
| 5 | `5_inferencia.ipynb` | Carregamento do artefato e processo de inferência |

As bases intermediárias estão armazenadas em:

```text
modelagem/bases_preparadas/
```

A divisão utilizada foi:

| Conjunto | Registros |
|---|---:|
| Treinamento | 1.460 |
| Validação | 313 |
| Teste | 314 |

## Modelo final

O modelo selecionado foi um **XGBoost multiclasse**.

| Informação | Resultado |
|---|---|
| Tipo de problema | Classificação multiclasse |
| Número de classes | 7 |
| Modelo final | XGBoost |
| Acurácia no teste holdout | 96,82% |
| Corte mínimo esperado | 75% |

O artefato final está armazenado em:

```text
artifacts/modelo_xgboost_obesidade.joblib
```

O arquivo contém:

- modelo treinado;
- colunas de entrada;
- melhores hiperparâmetros;
- valores utilizados no preenchimento de dados ausentes;
- importância das variáveis;
- métricas de treinamento, validação e teste;
- conjuntos utilizados na modelagem;
- mapeamento das classes.

## Features utilizadas pelo modelo

Após o processo de preparação, o modelo utiliza oito features:

- `Age`;
- `BMI`;
- `flag_female`;
- `flag_family_history`;
- `FCVC`;
- `CAEC_freq`;
- `CALC_freq`;
- `MTRANS_Public_Transportation`.

## Aplicação Streamlit

A aplicação possui três páginas principais.

### Predição individual

Permite informar os dados de uma pessoa e apresenta:

- classe estimada;
- probabilidade da classe prevista;
- BMI calculado;
- probabilidades das sete classes;
- features enviadas ao modelo.

### Predição em lote

Permite enviar um arquivo CSV com vários registros.

A aplicação:

- valida as colunas obrigatórias;
- trata valores ausentes;
- executa as previsões em lote;
- preserva as colunas originais;
- calcula o BMI;
- apresenta a classe prevista;
- informa a confiança da previsão;
- calcula as probabilidades das sete classes;
- disponibiliza o resultado completo para download.

A página também oferece um CSV de exemplo.

### Dashboard analítico

O projeto apresenta duas visões analíticas:

- dashboard da base utilizada na modelagem;
- dashboard criado a partir dos dados enviados na predição em lote.

Os painéis contêm indicadores, filtros e gráficos relacionados a:

- distribuição das classes;
- idade;
- BMI;
- gênero;
- histórico familiar;
- atividade física;
- confiança das previsões;
- desempenho do modelo;
- importância das features.

## Colunas necessárias para predição

Para realizar uma predição individual ou em lote, são utilizadas as seguintes colunas da base original:

```text
Age
Height
Weight
Gender
family_history
FCVC
CAEC
CALC
MTRANS
```

Colunas adicionais podem permanecer no CSV e serão preservadas no arquivo de resultado.

## Estrutura do repositório

```text
postech-fase4-g22/
├── app.py
├── requirements.txt
├── verificar_projeto.py
├── README.md
├── GUIA_RAPIDO.txt
├── 1_INSTALAR.bat
├── 2_TESTAR.bat
├── 3_INICIAR_APP.bat
│
├── pages/
│   ├── 1_Predicao.py
│   ├── 2_Predicao_em_Lote.py
│   └── 3_Dashboard.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model_service.py
│   └── preprocessing.py
│
├── artifacts/
│   └── modelo_xgboost_obesidade.joblib
│
├── data/
│   └── Obesity.csv
│
└── modelagem/
    ├── 1_exploracao.ipynb
    ├── 2_preparacao.ipynb
    ├── 3_modelagem.ipynb
    ├── 4_modelo_final.ipynb
    ├── 5_inferencia.ipynb
    │
    └── bases_preparadas/
        ├── X_train.csv
        ├── X_val.csv
        ├── X_test.csv
        ├── y_train.csv
        ├── y_val.csv
        └── y_test.csv
```

## Execução local simplificada no Windows

### 1. Baixar o projeto

No GitHub, selecione:

```text
Code → Download ZIP
```

Extraia o arquivo e entre na pasta do projeto.

### 2. Instalar

Execute:

```text
1_INSTALAR.bat
```

A primeira instalação pode levar alguns minutos.

### 3. Testar

Execute:

```text
2_TESTAR.bat
```

O resultado esperado deve conter:

```text
Projeto verificado com sucesso.
```

### 4. Iniciar a aplicação

Execute:

```text
3_INICIAR_APP.bat
```

A aplicação será aberta em:

```text
http://localhost:8501
```

## Execução pelo terminal

### Windows

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verificar_projeto.py
python -m streamlit run app.py
```

### macOS ou Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verificar_projeto.py
python -m streamlit run app.py
```

## Verificação automática

O arquivo `verificar_projeto.py` realiza as seguintes verificações:

- existência dos arquivos obrigatórios;
- carregamento do artefato;
- reprodução da acurácia no conjunto de teste;
- predição individual de exemplo;
- predição em lote;
- existência das probabilidades das sete classes;
- preservação da ordem das linhas.

Para executar:

```bash
python verificar_projeto.py
```

## Deploy no Streamlit Community Cloud

Para publicar a aplicação:

1. Acesse [share.streamlit.io](https://share.streamlit.io/).
2. Entre utilizando uma conta do GitHub.
3. Selecione **Create app**.
4. Escolha este repositório.
5. Utilize a branch `main`.
6. Informe `app.py` como arquivo principal.
7. Selecione Python 3.13 nas configurações avançadas.
8. Clique em **Deploy**.

Após o deploy, os links da aplicação e do dashboard devem ser adicionados no início deste README e no arquivo de links da entrega.

## Limitações

- A base é acadêmica e parcialmente sintética.
- O modelo classifica o estado apresentado no registro e não prevê obesidade futura.
- Os resultados não devem ser interpretados como diagnóstico médico.
- As associações exibidas nos dashboards não demonstram causalidade.
- A aplicação foi desenvolvida para fins acadêmicos e de apoio analítico.

## Tecnologias utilizadas

- Python 3.13;
- pandas;
- NumPy;
- scikit-learn;
- XGBoost;
- joblib;
- Plotly;
- Streamlit.
