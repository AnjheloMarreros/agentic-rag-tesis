import json
from pathlib import Path

def cargar_caso(caso_id: str):
    ruta = Path(f"data/casos/{caso_id}.json")
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)