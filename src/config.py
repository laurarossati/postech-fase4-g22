from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT_DIR / "artifacts" / "modelo_xgboost_obesidade.joblib"
DATA_PATH = ROOT_DIR / "data" / "Obesity.csv"

RAW_REQUIRED_COLUMNS = [
    "Age",
    "Height",
    "Weight",
    "Gender",
    "family_history",
    "FCVC",
    "CAEC",
    "CALC",
    "MTRANS",
]

CLASS_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

CLASS_LABELS = {
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso nível I",
    "Overweight_Level_II": "Sobrepeso nível II",
    "Obesity_Type_I": "Obesidade tipo I",
    "Obesity_Type_II": "Obesidade tipo II",
    "Obesity_Type_III": "Obesidade tipo III",
}

CLASS_COLORS = {
    "Peso insuficiente": "#60A5FA",
    "Peso normal": "#22C55E",
    "Sobrepeso nível I": "#FACC15",
    "Sobrepeso nível II": "#FB923C",
    "Obesidade tipo I": "#F87171",
    "Obesidade tipo II": "#EF4444",
    "Obesidade tipo III": "#B91C1C",
}

FREQUENCY_MAP = {
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3,
}

ALLOWED_CATEGORIES = {
    "Gender": {"Female", "Male"},
    "family_history": {"yes", "no"},
    "CAEC": set(FREQUENCY_MAP),
    "CALC": set(FREQUENCY_MAP),
    "MTRANS": {
        "Automobile",
        "Bike",
        "Motorbike",
        "Public_Transportation",
        "Walking",
    },
}

OBESITY_CLASSES = {
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
}
