from __future__ import annotations

import re
import unicodedata


def normalizar_texto(texto: str) -> str:
    """
    Normaliza texto para evaluación:
    - elimina espacios duplicados
    - unifica saltos de línea
    - conserva el contenido textual
    - no depende de PDF ni de librerías externas pesadas
    """
    if texto is None:
        return ""

    texto = str(texto)

    # Normalización Unicode
    texto = unicodedata.normalize("NFKC", texto)

    # Unifica saltos de línea
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")

    # Reduce espacios y tabs repetidos, pero preserva estructura básica
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n\s+\n", "\n\n", texto)

    return texto.strip()