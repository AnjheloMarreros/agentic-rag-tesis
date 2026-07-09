from __future__ import annotations

import re
import unicodedata

def normalizar_texto(texto: str | None) -> str:

    if not texto:
        return ""

    texto = str(texto).strip()
    
    texto = unicodedata.normalize("NFKC", texto)

    for caracter in ("“", "”", "«", "»", "„", "‟", '"', "'"):
        texto = texto.replace(caracter, " ")

    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()