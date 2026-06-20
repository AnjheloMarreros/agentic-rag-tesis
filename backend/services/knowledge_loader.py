from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = BASE_DIR / "data" / "docs"


def leer_txt(ruta: Path) -> str:
    with open(ruta, "r", encoding="utf-8") as archivo:
        return archivo.read().strip()


def cargar_documentos() -> List[Dict[str, str]]:
    documentos: List[Dict[str, str]] = []

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de documentos: {DOCS_DIR}"
        )

    for archivo in DOCS_DIR.iterdir():
        if not archivo.is_file():
            continue

        if archivo.suffix.lower() != ".txt":
            continue

        texto = leer_txt(archivo)
        if not texto:
            continue

        documentos.append(
            {
                "source": archivo.stem,
                "type": "txt",
                "text": texto,
            }
        )

    if not documentos:
        raise ValueError(
            "No se encontraron documentos .txt para cargar en la base vectorial."
        )

    return documentos