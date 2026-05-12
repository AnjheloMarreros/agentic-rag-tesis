from datetime import datetime
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

ARCHIVO_LOG = LOGS_DIR / "eventos.jsonl"


def registrar_evento(tipo: str, datos: dict):
    evento = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tipo": tipo,
        "datos": datos
    }

    with open(ARCHIVO_LOG, "a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(evento, ensure_ascii=False) + "\n")