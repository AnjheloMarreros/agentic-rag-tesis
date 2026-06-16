from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader


def normalizar_texto(texto: str) -> str:
    if not texto:
        return ""

    texto = unicodedata.normalize("NFKC", str(texto))

    # Quita comillas sin borrar el contenido.
    for caracter in ("“", "”", "«", "»", "„", "‟", '"', "'"):
        texto = texto.replace(caracter, " ")

    # Normaliza saltos de línea y espacios.
    texto = texto.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def extraer_texto_pdf(ruta_pdf: str | Path) -> str:
    """
    Se conserva por compatibilidad, aunque ya no usarás PDF en el flujo.
    """
    reader = PdfReader(str(ruta_pdf))
    partes: list[str] = []

    for page in reader.pages:
        texto = page.extract_text() or ""
        if texto.strip():
            partes.append(texto)

    return normalizar_texto("\n".join(partes))