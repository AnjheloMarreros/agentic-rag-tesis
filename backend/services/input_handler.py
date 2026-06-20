from __future__ import annotations

import re
import unicodedata

def normalizar_texto(texto: str | None) -> str:
    """
    Normaliza el texto eliminando comillas, estandarizando espacios 
    y aplicando normalización Unicode.
    """
    if not texto:
        return ""

    # 1. Aseguramos que sea string y limpiamos espacios a los lados (de la propuesta)
    texto = str(texto).strip()
    
    # 2. Normalización Unicode
    texto = unicodedata.normalize("NFKC", texto)

    # 3. Mantenemos tu lógica original: Quita comillas reemplazándolas por un espacio
    for caracter in ("“", "”", "«", "»", "„", "‟", '"', "'"):
        texto = texto.replace(caracter, " ")

    # 4. Usamos la mejora propuesta para los saltos de línea y espacios extras
    # re.sub(r"\s+", " ", texto) ya se encarga de los \n, \r, \t y espacios múltiples
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()