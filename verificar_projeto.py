import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.config import ARTIFACT_PATH, DATA_PATH, ROOT_DIR
from src.model_service import load_artifact, predict_batch, predict_patient


def main() -> None:
    required_paths = [
        ROOT_DIR / "app.py",
        ROOT_DIR / "pages/1_Predicao.py",
        ROOT_DIR / "pages/2_Predicao_em_Lote.py",
        ROOT_DIR / "pages/3_Dashboard.py",
        ARTIFACT_PATH,
        DATA_PATH,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    assert not missing_paths, f"Arquivos ausentes: {missing_paths}"

    artifact = load_artifact()
    test_x = artifact["dados"]["X_test"]
    test_y = np.asarray(artifact["dados"]["y_test"]).ravel()
    test_prediction = artifact["modelo"].predict(test_x)
    test_accuracy = accuracy_score(test_y, test_prediction)
    assert test_accuracy >= 0.75, "Acurácia do teste abaixo de 75%."

    source_data = pd.read_csv(DATA_PATH)
    assert not source_data.empty, "Obesity.csv está vazio."

    sample = source_data.iloc[[0]].drop(columns=["Obesity"])
    result = predict_patient(sample)
    assert result["classe"], "A predição de exemplo não retornou uma classe."
    assert len(result["probabilidades"]) == 7

    batch = predict_batch(source_data.head(25))
    batch_results = batch["resultados"]
    assert len(batch_results) == 25
    assert batch_results["linha_csv"].tolist() == list(range(1, 26))
    assert "Obesity" in batch_results.columns
    assert "classe_predita" in batch_results.columns
    assert "probabilidade_predita_pct" in batch_results.columns
    probability_columns = [
        column
        for column in batch_results.columns
        if column.startswith("prob_")
    ]
    assert len(probability_columns) == 7
    probability_total = batch_results[probability_columns].sum(axis=1)
    assert np.allclose(probability_total, 100, atol=0.01)
    assert batch_results.loc[0, "classe_predita"] == result["classe"]

    print("Projeto verificado com sucesso.")
    print(f"Acurácia reproduzida no teste: {test_accuracy:.2%}")
    print(f"Predição de exemplo: {result['classe']}")
    print("Predição em lote: 25 registros e 7 probabilidades validados.")


if __name__ == "__main__":
    main()
