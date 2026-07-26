from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    ALLOWED_CATEGORIES,
    FREQUENCY_MAP,
    RAW_REQUIRED_COLUMNS,
)


def prepare_model_input(
    raw_data: pd.DataFrame,
    input_columns: list[str],
    fill_values: dict,
) -> pd.DataFrame:
    """Transforma dados brutos nas oito features esperadas pelo modelo."""
    if raw_data.empty:
        raise ValueError("Nenhum registro foi informado.")

    df = raw_data.copy()
    df.columns = df.columns.str.strip()

    duplicated_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(
            "Existem colunas duplicadas no CSV: "
            + ", ".join(duplicated_columns)
        )

    missing_columns = [
        column for column in RAW_REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(missing_columns)
        )

    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].replace("", pd.NA)

    df = df.fillna(value=fill_values)

    null_columns = [
        column
        for column in RAW_REQUIRED_COLUMNS
        if df[column].isna().any()
    ]
    if null_columns:
        raise ValueError(
            "Ainda existem valores ausentes em: " + ", ".join(null_columns)
        )

    for column, allowed_values in ALLOWED_CATEGORIES.items():
        invalid_values = sorted(set(df[column].dropna()) - allowed_values)
        if invalid_values:
            raise ValueError(
                f"Valores inválidos em {column}: {invalid_values}"
            )

    for column in ["Age", "Height", "Weight", "FCVC"]:
        normalized_values = df[column].astype("string").str.replace(
            ",",
            ".",
            regex=False,
        )
        df[column] = pd.to_numeric(normalized_values, errors="coerce")

    if df[["Age", "Height", "Weight", "FCVC"]].isna().any().any():
        raise ValueError("Idade, altura, peso e FCVC devem ser numéricos.")

    if (df["Height"] <= 0).any() or (df["Weight"] <= 0).any():
        raise ValueError("Altura e peso devem ser maiores que zero.")

    df["FCVC"] = df["FCVC"].round().astype(int)
    df["flag_female"] = df["Gender"].map({"Female": 1, "Male": 0})
    df["flag_family_history"] = df["family_history"].map(
        {"yes": 1, "no": 0}
    )
    df["CAEC_freq"] = df["CAEC"].map(FREQUENCY_MAP)
    df["CALC_freq"] = df["CALC"].map(FREQUENCY_MAP)
    df["MTRANS_Public_Transportation"] = (
        df["MTRANS"] == "Public_Transportation"
    ).astype(int)
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    model_input = df.reindex(columns=input_columns)
    model_input = model_input.replace([np.inf, -np.inf], np.nan)

    null_model_columns = model_input.columns[
        model_input.isna().any()
    ].tolist()
    if null_model_columns:
        raise ValueError(
            "Não foi possível preparar as features: "
            + ", ".join(null_model_columns)
        )

    if model_input.columns.tolist() != input_columns:
        raise ValueError("A ordem das features não corresponde ao treinamento.")

    return model_input
