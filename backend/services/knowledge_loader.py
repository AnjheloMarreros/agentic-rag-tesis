from pathlib import Path
from typing import List, Dict, Iterable

BASE_DIR = Path(__file__).resolve().parents[2]

KNOWLEDGE_DIR = BASE_DIR / "data" / "docs" / "knowledge"
PEDAGOGICAL_DIR = BASE_DIR / "data" / "docs" / "pedagogical"
DOCS_FALLBACK_DIR = BASE_DIR / "data" / "docs"

ALLOWED_EXTENSIONS = {".txt", ".md"}


def leer_texto(ruta: Path) -> str:
    with open(ruta, "r", encoding="utf-8") as archivo:
        return archivo.read().strip()


def _iterar_archivos_documento(directorios: Iterable[Path]):
    vistos = set()

    for directorio in directorios:
        if not directorio.exists():
            continue

        for archivo in directorio.rglob("*"):
            if not archivo.is_file():
                continue

            if archivo.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            ruta_resuelta = archivo.resolve()
            if ruta_resuelta in vistos:
                continue

            vistos.add(ruta_resuelta)
            yield ruta_resuelta


def cargar_documentos(include_pedagogical: bool = False) -> List[Dict[str, str]]:
    directorios = [KNOWLEDGE_DIR]

    if include_pedagogical:
        directorios.append(PEDAGOGICAL_DIR)

    directorios.append(DOCS_FALLBACK_DIR)

    documentos: List[Dict[str, str]] = []

    archivos = list(_iterar_archivos_documento(directorios))

    if not archivos:
        raise ValueError(
            "No se encontraron documentos .txt ni .md para cargar en la base vectorial."
        )

    for archivo in sorted(archivos):
        texto = leer_texto(archivo)
        if not texto:
            continue

        if KNOWLEDGE_DIR in archivo.parents:
            scope = "knowledge"
        elif PEDAGOGICAL_DIR in archivo.parents:
            scope = "pedagogical"
        else:
            scope = "fallback"

        documentos.append(
            {
                "source": archivo.relative_to(BASE_DIR).as_posix(),
                "type": archivo.suffix.lower().lstrip("."),
                "text": texto,
                "scope": scope,
            }
        )

    if not documentos:
        raise ValueError(
            "Se encontraron archivos de texto, pero todos estaban vacíos."
        )

    return documentos