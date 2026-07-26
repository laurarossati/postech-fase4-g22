from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACT_PATH, CLASS_LABELS
from src.preprocessing import prepare_model_input


@lru_cache(maxsize=1)
def load_artifact(path: Path = ARTIFACT_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Artefato não encontrado: {path}")

    artifact = joblib.load(path)
    expected_keys = {
        "modelo",
        "target_map",
        "colunas_input",
        "valores_fillna",
        "feature_importance",
        "metricas_treinamento",
        "dados",
    }
    missing_keys = expected_keys - set(artifact)
    if missing_keys:
        raise ValueError(
            "Chaves ausentes no artefato: " + ", ".join(sorted(missing_keys))
        )
    return artifact


def predict_patient(raw_data: pd.DataFrame) -> dict:
    artifact = load_artifact()
    model = artifact["modelo"]

    model_input = prepare_model_input(
        raw_data=raw_data,
        input_columns=artifact["colunas_input"],
        fill_values=artifact["valores_fillna"],
    )

    predicted_code = int(model.predict(model_input)[0])
    probabilities = model.predict_proba(model_input)[0]
    predicted_position = list(model.classes_).index(predicted_code)

    inverse_target_map = {
        code: name for name, code in artifact["target_map"].items()
    }
    predicted_name = inverse_target_map[predicted_code]

    probability_table = pd.DataFrame(
        {
            "codigo": model.classes_.astype(int),
            "classe_original": [
                inverse_target_map[int(code)] for code in model.classes_
            ],
            "probabilidade": probabilities,
        }
    )
    probability_table["classe"] = probability_table["classe_original"].map(
        CLASS_LABELS
    )
    probability_table["probabilidade_pct"] = (
        probability_table["probabilidade"] * 100
    )

    return {
        "codigo": predicted_code,
        "classe_original": predicted_name,
        "classe": CLASS_LABELS[predicted_name],
        "probabilidade": float(probabilities[predicted_position]),
        "probabilidades": probability_table,
        "model_input": model_input,
    }


def predict_batch(raw_data: pd.DataFrame) -> dict:
    """Realiza a predição vetorizada e preserva uma linha por registro."""
    artifact = load_artifact()
    model = artifact["modelo"]

    model_input = prepare_model_input(
        raw_data=raw_data,
        input_columns=artifact["colunas_input"],
        fill_values=artifact["valores_fillna"],
    )

    predicted_codes = model.predict(model_input).astype(int)
    probabilities = model.predict_proba(model_input)
    inverse_target_map = {
        int(code): name
        for name, code in artifact["target_map"].items()
    }
    predicted_names = [
        inverse_target_map[int(code)] for code in predicted_codes
    ]
    class_positions = {
        int(code): position
        for position, code in enumerate(model.classes_)
    }
    predicted_probabilities = np.array(
        [
            probabilities[row, class_positions[int(code)]]
            for row, code in enumerate(predicted_codes)
        ]
    )

    results = raw_data.copy()
    results.columns = results.columns.str.strip()
    results = results.reset_index(drop=True)
    results.insert(0, "linha_csv", np.arange(1, len(results) + 1))
    results["BMI_calculado"] = model_input["BMI"].to_numpy()
    results["classe_predita_codigo"] = predicted_codes
    results["classe_predita_original"] = predicted_names
    results["classe_predita"] = [
        CLASS_LABELS[name] for name in predicted_names
    ]
    results["probabilidade_predita_pct"] = (
        predicted_probabilities * 100
    )

    for position, code in enumerate(model.classes_):
        class_name = inverse_target_map[int(code)]
        results[f"prob_{class_name}_pct"] = (
            probabilities[:, position] * 100
        )

    return {
        "resultados": results,
        "model_input": model_input.reset_index(drop=True),
    }
