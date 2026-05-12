from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[2]
CASOS_DIR = BASE_DIR / "data" / "casos"


def cargar_caso(caso_id: str):
    ruta = CASOS_DIR / f"{caso_id}.json"

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    with open(ruta, "r", encoding="utf-8") as archivo:
        return json.load(archivo)