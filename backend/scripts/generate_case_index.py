from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
CASOS_DIR = BASE_DIR / "data" / "casos"
KNOWLEDGE_DIR = BASE_DIR / "data" / "docs" / "knowledge"
OUTPUT_FILE = BASE_DIR / "data" / "casos_index.json"


PALABRAS_JURIDICAS = [
    "artículo", "articulo", "código penal", "codigo penal", "pena", "discriminación",
    "discriminacion", "igualdad", "proporcionalidad", "razonabilidad", "justificación",
    "justificacion", "excepción", "excepcion", "responsabilidad restringida",
    "robo agravado", "violencia", "juez", "legislador", "derecho", "norma", "ley",
]

def _leer_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} no contiene un objeto JSON válido.")
    return data

def _leer_md(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")

def _extraer_frases_clave(texto: str, min_len: int = 4) -> list[str]:
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñü0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    candidatos: list[str] = []
    for patron in PALABRAS_JURIDICAS:
        if patron in texto:
            candidatos.append(patron)

    # frases de 2 a 4 palabras que contengan términos jurídicos
    palabras = texto.split()
    for n in range(2, 5):
        for i in range(len(palabras) - n + 1):
            frase = " ".join(palabras[i:i+n])
            if any(p in frase for p in PALABRAS_JURIDICAS):
                candidatos.append(frase)

    # depurar repetidos
    unicos = list(dict.fromkeys([c.strip() for c in candidatos if len(c.strip()) >= min_len]))
    return unicos[:20]

def generar_indice() -> None:
    indice: list[dict[str, Any]] = []

    for archivo_json in sorted(CASOS_DIR.glob("*.json")):
        caso = _leer_json(archivo_json)
        case_id = str(caso.get("id", archivo_json.stem)).strip()
        titulo = str(caso.get("titulo", archivo_json.stem)).strip()

        md_path = KNOWLEDGE_DIR / f"{case_id}.md"
        md_texto = _leer_md(md_path)

        frases = _extraer_frases_clave(md_texto)

        # Heurística simple: repartir frases entre hechos, normas y conceptos.
        hechos = frases[:4]
        normas = [f for f in frases if "artículo" in f or "articulo" in f or "código penal" in f or "codigo penal" in f][:4]
        conceptos = [f for f in frases if f not in hechos][:6]

        indice.append(
            {
                "case_id": case_id,
                "titulo": titulo,
                "knowledge_path": str(md_path.relative_to(BASE_DIR)),
                "hechos_clave": hechos,
                "normas_clave": normas,
                "conceptos_esperados": conceptos,
            }
        )

    OUTPUT_FILE.write_text(
        json.dumps(indice, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

if __name__ == "__main__":
    generar_indice()