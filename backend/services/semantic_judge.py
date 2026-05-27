from __future__ import annotations

from functools import lru_cache
from typing import Any
import re
import unicodedata
from math import sqrt

import numpy as np
from sentence_transformers import SentenceTransformer


CONECTORES = [
    "porque",
    "por tanto",
    "sin embargo",
    "en consecuencia",
    "además",
    "por ello",
    "por consiguiente",
    "ya que",
    "puesto que",
    "por otra parte",
    "no obstante",
]

MARCADORES_CONCLUSION = [
    "por tanto",
    "por lo tanto",
    "en consecuencia",
    "por ello",
    "por consiguiente",
    "concluyo",
    "conclusion",
    "conclusión",
]

FRASES_DESCONOCIMIENTO = [
    "no estoy enterado",
    "no se",
    "no sé",
    "desconozco",
    "no tengo idea",
    "no conozco",
]

TERMINOS_JURIDICOS = [
    "derecho",
    "norma",
    "constitucion",
    "constitución",
    "proceso",
    "tribunal",
    "demanda",
    "identidad",
    "caducidad",
    "control difuso",
    "codigo civil",
    "código civil",
    "constitucional",
    "supranacional",
    "convencion",
    "convención",
    "argumentacion",
    "argumentación",
]


@lru_cache(maxsize=1)
def _modelo():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _normalizar(texto: str) -> str:
    if not texto:
        return ""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9ñü\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def _palabras(texto: str) -> list[str]:
    return [t for t in _normalizar(texto).split() if len(t) > 2]


def _vector(texto: str):
    texto = _normalizar(texto)
    if not texto:
        return None
    try:
        return _modelo().encode([texto], normalize_embeddings=True)[0]
    except Exception:
        return None


def _coseno(a, b) -> float:
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _clamp(valor: float, minimo: int = 1, maximo: int = 5) -> int:
    return max(minimo, min(maximo, int(round(valor))))


def _nivel(puntaje: int) -> str:
    if puntaje <= 1:
        return "Muy bajo"
    if puntaje == 2:
        return "Bajo"
    if puntaje == 3:
        return "Medio"
    if puntaje == 4:
        return "Alto"
    return "Excelente"


def _contiene_alguna(texto: str, lista: list[str]) -> bool:
    texto_n = _normalizar(texto)
    return any(item in texto_n for item in lista)


def _contar_terminos_juridicos(texto: str) -> int:
    texto_n = _normalizar(texto)
    return sum(1 for termino in TERMINOS_JURIDICOS if termino in texto_n)


def _top_fragmento(respuesta: str, contexto_recuperado: list[dict[str, Any]]) -> str:
    if not contexto_recuperado:
        return ""

    resp_vec = _vector(respuesta)
    mejor = ""
    mejor_score = -1.0

    for item in contexto_recuperado:
        fragmento = item.get("fragmento", "") if isinstance(item, dict) else ""
        if not fragmento:
            continue

        frag_vec = _vector(fragmento)
        sim = _coseno(resp_vec, frag_vec)

        if sim > mejor_score:
            mejor_score = sim
            mejor = fragmento

    return mejor


def _puntaje_desde_similitud(similitud: float) -> int:
    if similitud >= 0.82:
        return 5
    if similitud >= 0.66:
        return 4
    if similitud >= 0.50:
        return 3
    if similitud >= 0.32:
        return 2
    return 1


def evaluar_semantica(
    respuesta: str,
    caso: dict[str, Any],
    contexto_recuperado: list[dict[str, Any]],
) -> dict[str, Any]:
    respuesta = respuesta or ""
    palabras = _palabras(respuesta)
    n_palabras = len(palabras)

    texto_caso = " ".join([
        caso.get("titulo", ""),
        caso.get("enunciado", ""),
        " ".join(caso.get("contexto", [])),
        " ".join(caso.get("instrucciones", [])),
    ]).strip()

    texto_contexto = " ".join(
        item.get("fragmento", "")
        for item in contexto_recuperado
        if isinstance(item, dict)
    ).strip()

    vec_respuesta = _vector(respuesta)
    vec_caso = _vector(texto_caso)
    vec_contexto = _vector(texto_contexto)

    similitud_caso = _coseno(vec_respuesta, vec_caso)
    similitud_contexto = _coseno(vec_respuesta, vec_contexto)

    terminos_juridicos = _contar_terminos_juridicos(respuesta)
    conectores = sum(1 for c in CONECTORES if c in _normalizar(respuesta))
    conclusion = _contiene_alguna(respuesta, MARCADORES_CONCLUSION)
    desconocimiento = _contiene_alguna(respuesta, FRASES_DESCONOCIMIENTO)

    # 1) Pertinencia con el caso
    score_caso = 1 + (similitud_caso * 3.0) + min(terminos_juridicos, 2) * 0.5
    if conclusion:
        score_caso += 0.25
    score_caso = _clamp(score_caso)
    obs_caso = (
        "La respuesta guarda relación semántica con el caso."
        if score_caso >= 4 else
        "La relación con el caso todavía es limitada."
    )
    rec_caso = (
        "Mantén el foco en el problema jurídico planteado."
        if score_caso < 4 else
        "La respuesta está bien alineada con el caso."
    )

    # 2) Sustento contextual
    score_contexto = 1 + (similitud_contexto * 3.0) + min(terminos_juridicos, 2) * 0.5
    if conectores >= 2:
        score_contexto += 0.25
    score_contexto = _clamp(score_contexto)
    obs_contexto = (
        "La respuesta se apoya en parte del contexto recuperado."
        if score_contexto >= 4 else
        "La respuesta usa poco el contexto recuperado."
    )
    rec_contexto = (
        "Vincula explícitamente tus ideas con el contexto jurídico recuperado."
        if score_contexto < 4 else
        "El sustento contextual es adecuado."
    )

    # 3) Profundidad argumentativa
    score_argumento = 1
    if n_palabras >= 20:
        score_argumento += 1
    if n_palabras >= 45:
        score_argumento += 1
    if conectores >= 2:
        score_argumento += 1
    if conclusion:
        score_argumento += 1
    if n_palabras >= 80:
        score_argumento += 1
    if desconocimiento and n_palabras < 12:
        score_argumento = 1
    score_argumento = _clamp(score_argumento)
    obs_argumento = (
        "La respuesta muestra un desarrollo argumentativo aceptable."
        if score_argumento >= 4 else
        "La respuesta todavía necesita más desarrollo argumentativo."
    )
    rec_argumento = (
        "Amplía la respuesta con premisas, análisis y conclusión."
        if score_argumento < 4 else
        "El desarrollo argumentativo es sólido."
    )

    # 4) Consistencia semántica
    inconsistencia_penalizacion = 0
    if desconocimiento and n_palabras < 12:
        inconsistencia_penalizacion += 2
    if "pero" in _normalizar(respuesta) and "sin embargo" not in _normalizar(respuesta):
        inconsistencia_penalizacion += 1

    score_consistencia = 3 + (similitud_caso * 1.5) + (similitud_contexto * 1.0) - inconsistencia_penalizacion
    score_consistencia = _clamp(score_consistencia)

    obs_consistencia = (
        "La respuesta mantiene una coherencia semántica razonable."
        if score_consistencia >= 4 else
        "La coherencia semántica aún puede fortalecerse."
    )
    rec_consistencia = (
        "Revisa si tus afirmaciones se sostienen entre sí."
        if score_consistencia < 4 else
        "La consistencia semántica es adecuada."
    )

    criterios = [
        {
            "clave": "pertinencia_caso",
            "nombre": "Pertinencia con el caso",
            "puntaje": score_caso,
            "nivel": _nivel(score_caso),
            "observacion": obs_caso,
            "recomendacion": rec_caso,
        },
        {
            "clave": "sustento_contextual",
            "nombre": "Sustento contextual",
            "puntaje": score_contexto,
            "nivel": _nivel(score_contexto),
            "observacion": obs_contexto,
            "recomendacion": rec_contexto,
        },
        {
            "clave": "profundidad_argumentativa",
            "nombre": "Profundidad argumentativa",
            "puntaje": score_argumento,
            "nivel": _nivel(score_argumento),
            "observacion": obs_argumento,
            "recomendacion": rec_argumento,
        },
        {
            "clave": "consistencia_semantica",
            "nombre": "Consistencia semántica",
            "puntaje": score_consistencia,
            "nivel": _nivel(score_consistencia),
            "observacion": obs_consistencia,
            "recomendacion": rec_consistencia,
        },
    ]

    puntaje_total_0_5 = sum(c["puntaje"] for c in criterios) / len(criterios)
    puntaje_total = round((puntaje_total_0_5 / 5) * 100, 1)

    if puntaje_total >= 85:
        nivel_global = "Excelente"
    elif puntaje_total >= 70:
        nivel_global = "Alto"
    elif puntaje_total >= 55:
        nivel_global = "Medio"
    else:
        nivel_global = "Bajo"

    observaciones = [c["observacion"] for c in criterios]
    recomendaciones = [c["recomendacion"] for c in criterios]

    evidencia = _top_fragmento(respuesta, contexto_recuperado)

    return {
        "puntaje_total": puntaje_total,
        "nivel_global": nivel_global,
        "resumen": f"Tu respuesta obtuvo {puntaje_total}% de coherencia semántica. El nivel global es {nivel_global}.",
        "similitud_caso": round(similitud_caso, 4),
        "similitud_contexto": round(similitud_contexto, 4),
        "evidencia_principal": evidencia,
        "criterios": criterios,
        "observaciones": observaciones,
        "recomendaciones": recomendaciones,
    }