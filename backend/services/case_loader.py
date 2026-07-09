from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
CASOS_DIR = BASE_DIR / "data" / "casos"


def listar_casos() -> list[dict[str, str]]:
    if not CASOS_DIR.exists():
        return []

    casos: list[dict[str, str]] = []

    for archivo in sorted(CASOS_DIR.glob("*.json")):
        try:
            with open(archivo, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                continue

            casos.append(
                {
                    "id": str(data.get("id", archivo.stem)),
                    "titulo": str(data.get("titulo", archivo.stem)),
                }
            )
        except Exception:
            continue

    return casos


def cargar_caso(caso_id: str) -> dict[str, Any]:
    ruta = CASOS_DIR / f"{caso_id}.json"

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    with open(ruta, "r", encoding="utf-8") as archivo:
        data = json.load(archivo)

    if not isinstance(data, dict):
        raise ValueError(f"El archivo {ruta} no contiene un objeto JSON válido.")

    return data