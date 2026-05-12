from fastapi import FastAPI
import json
from pathlib import Path

app = FastAPI()

@app.get("/caso")
def obtener_caso():
    ruta = Path("data/casos/caso_001.json")
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)