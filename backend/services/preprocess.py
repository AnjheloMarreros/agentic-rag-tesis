from __future__ import annotations

import re
import unicodedata


def normalizar_texto(texto: str) -> str:
    """
    Normaliza texto plano para evaluación.
    No depende de PDF ni de otras librerías externas pesadas.
    """
    if texto is None:
        return ""

    texto = str(texto)

    # Normalización Unicode
    texto = unicodedata.normalize("NFKC", texto)

    # Unifica saltos de línea
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")

    # Reduce espacios y tabs repetidos
    texto = re.sub(r"[ \t]+", " ", texto)

    # Limpia líneas con solo espacios
    texto = re.sub(r"\n\s+\n", "\n\n", texto)

    return texto.strip()