from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[2]
RUTA_RUBRICA = BASE_DIR / "data" / "rubrics" / "rubrica.json"


def cargar_rubrica():
    if not RUTA_RUBRICA.exists():
        raise FileNotFoundError(
            f"No existe la rúbrica: {RUTA_RUBRICA}"
        )

    with open(RUTA_RUBRICA, "r", encoding="utf-8") as archivo:
        return json.load(archivo)