from __future__ import annotations

from typing import Any
import re
import unicodedata


CONECTORES = [
    "porque",
    "por tanto",
    "sin embargo",
    "en consecuencia",
    "además",
    "por ello",
    "por consiguiente",
    "por lo tanto",
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

MARCADORES_PREMISA = [
    "premisa",
    "premisa mayor",
    "premisa menor",
    "razonamiento",
    "fundamento",
]

FRASES_DESCONOCIMIENTO = [
    "no estoy enterado",
    "no se",
    "no sé",
    "desconozco",
    "no tengo idea",
    "no conozco",
]

STOPWORDS = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "u",
    "a", "ante", "bajo", "con", "contra", "desde", "durante", "entre", "hacia",
    "hasta", "para", "por", "segun", "según", "sin", "sobre", "tras", "del",
    "al", "que", "se", "es", "son", "ser", "como", "más", "menos", "muy",
    "en", "su", "sus", "le", "les", "lo", "ya", "no", "sí", "también",
    "caso", "casos", "respuesta", "argumento", "argumentar", "texto",
}


def _normalizar(texto: str) -> str:
    if not texto:
        return ""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9ñü\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def _tokens(texto: str) -> list[str]:
    texto_n = _normalizar(texto)
    tokens = [t for t in texto_n.split() if len(t) > 2]
    return [t for t in tokens if t not in STOPWORDS]


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


def _clamp(valor: float, minimo: int = 1, maximo: int = 5) -> int:
    return max(minimo, min(maximo, int(round(valor))))


def _contiene_alguna(texto: str, lista: list[str]) -> bool:
    texto_n = _normalizar(texto)
    return any(item in texto_n for item in lista)


def _texto_referencia(caso: dict[str, Any] | None, contexto_recuperado: list[dict[str, Any]]) -> str:
    partes: list[str] = []

    if isinstance(caso, dict):
        partes.append(str(caso.get("titulo", "")))
        partes.append(str(caso.get("enunciado", "")))
        partes.append(" ".join(caso.get("contexto", []) or []))
        partes.append(" ".join(caso.get("instrucciones", []) or []))

    for item in contexto_recuperado:
        if isinstance(item, dict):
            partes.append(str(item.get("fragmento", "")))

    return " ".join(partes).strip()


def _fragmento_mas_relevante(
    respuesta: str,
    contexto_recuperado: list[dict[str, Any]],
) -> str:
    if not contexto_recuperado:
        return ""

    resp_tokens = set(_tokens(respuesta))
    mejor_fragmento = ""
    mejor_score = 0.0

    for item in contexto_recuperado:
        if not isinstance(item, dict):
            continue

        fragmento = str(item.get("fragmento", "")).strip()
        if not fragmento:
            continue

        frag_tokens = set(_tokens(fragmento))
        if not frag_tokens:
            continue

        score = len(resp_tokens & frag_tokens) / len(frag_tokens)
        if score > mejor_score:
            mejor_score = score
            mejor_fragmento = fragmento

    return mejor_fragmento


def _palabras_clave_caso(caso: dict[str, Any]) -> list[str]:
    texto = " ".join(
        [
            str(caso.get("titulo", "")),
            str(caso.get("enunciado", "")),
            " ".join(caso.get("contexto", []) or []),
            " ".join(caso.get("instrucciones", []) or []),
        ]
    )
    tokens = _tokens(texto)
    claves = [tok for tok in tokens if len(tok) >= 6]
    return list(dict.fromkeys(claves))


def _indice_relevancia_lexica(respuesta: str, palabras_clave: list[str]) -> float:
    resp = set(_tokens(respuesta))
    keys = set(palabras_clave)
    if not keys or not resp:
        return 0.0
    overlap = resp.intersection(keys)
    return len(overlap) / len(keys)


def _cobertura_referencia(respuesta: str, texto_referencia: str) -> float:
    resp = set(_tokens(respuesta))
    ref = set(_tokens(texto_referencia))
    if not resp or not ref:
        return 0.0
    return len(resp.intersection(ref)) / len(ref)


def evaluar_semantica(
    respuesta: str,
    caso: dict[str, Any],
    contexto_recuperado: list[dict[str, Any]],
) -> dict[str, Any]:
    respuesta = respuesta or ""
    texto_n = _normalizar(respuesta)
    palabras = _tokens(respuesta)
    n_palabras = len(palabras)

    texto_caso = " ".join(
        [
            str(caso.get("titulo", "")),
            str(caso.get("enunciado", "")),
            " ".join(caso.get("contexto", []) or []),
            " ".join(caso.get("instrucciones", []) or []),
        ]
    ).strip()

    texto_contexto = " ".join(
        str(item.get("fragmento", ""))
        for item in contexto_recuperado
        if isinstance(item, dict)
    ).strip()

    palabras_clave = _palabras_clave_caso(caso)
    indice_lexico = _indice_relevancia_lexica(respuesta, palabras_clave)

    similitud_caso = _cobertura_referencia(respuesta, texto_caso)
    similitud_contexto = _cobertura_referencia(respuesta, texto_contexto)

    if indice_lexico < 0.10:
        indice_relevancia_caso = min((0.6 * similitud_caso) + (0.4 * indice_lexico), 0.15)
    elif indice_lexico < 0.20:
        indice_relevancia_caso = min((0.6 * similitud_caso) + (0.4 * indice_lexico), 0.25)
    elif indice_lexico < 0.30:
        indice_relevancia_caso = min((0.6 * similitud_caso) + (0.4 * indice_lexico), 0.35)
    else:
        indice_relevancia_caso = (0.6 * similitud_caso) + (0.4 * indice_lexico)

    conectores = sum(1 for c in CONECTORES if c in texto_n)
    conclusion = _contiene_alguna(texto_n, MARCADORES_CONCLUSION)
    premisas = _contiene_alguna(texto_n, MARCADORES_PREMISA)
    desconocimiento = _contiene_alguna(texto_n, FRASES_DESCONOCIMIENTO)

    score_caso = 1 + (indice_relevancia_caso * 4.0)
    score_contexto = 1 + (max(similitud_contexto, indice_lexico) * 4.0)

    if conclusion:
        score_caso += 0.25
        score_contexto += 0.15

    score_caso = _clamp(score_caso)
    score_contexto = _clamp(score_contexto)

    obs_caso = (
        "La respuesta guarda relación semántica con el caso."
        if score_caso >= 4
        else "La relación con el caso todavía es limitada."
    )
    rec_caso = (
        "La respuesta está bien alineada con el caso."
        if score_caso >= 4
        else "Mantén el foco en el problema jurídico planteado."
    )

    obs_contexto = (
        "La respuesta se apoya en parte del contexto recuperado."
        if score_contexto >= 4
        else "La respuesta usa poco el contexto recuperado."
    )
    rec_contexto = (
        "El sustento contextual es adecuado."
        if score_contexto >= 4
        else "Vincula explícitamente tus ideas con el contexto jurídico recuperado."
    )

    score_argumento = 1
    if n_palabras >= 20:
        score_argumento += 1
    if n_palabras >= 45:
        score_argumento += 1
    if conectores >= 2:
        score_argumento += 1
    if conclusion:
        score_argumento += 1
    if premisas:
        score_argumento += 1

    if desconocimiento and n_palabras < 12:
        score_argumento = 1

    score_argumento = _clamp(score_argumento)
    obs_argumento = (
        "La respuesta muestra un desarrollo argumentativo aceptable."
        if score_argumento >= 4
        else "La respuesta todavía necesita más desarrollo argumentativo."
    )
    rec_argumento = (
        "El desarrollo argumentativo es sólido."
        if score_argumento >= 4
        else "Amplía la respuesta con premisas, análisis y conclusión."
    )

    inconsistencia_penalizacion = 0
    if desconocimiento and n_palabras < 12:
        inconsistencia_penalizacion += 2
    if "pero" in texto_n and "sin embargo" not in texto_n:
        inconsistencia_penalizacion += 1

    score_consistencia = 3 + (similitud_caso * 1.0) + (similitud_contexto * 0.75) - inconsistencia_penalizacion
    score_consistencia = _clamp(score_consistencia)

    obs_consistencia = (
        "La respuesta mantiene una coherencia semántica razonable."
        if score_consistencia >= 4
        else "La coherencia semántica aún puede fortalecerse."
    )
    rec_consistencia = (
        "La consistencia semántica es adecuada."
        if score_consistencia >= 4
        else "Revisa si tus afirmaciones se sostienen entre sí."
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

    evidencia = _fragmento_mas_relevante(respuesta, contexto_recuperado)

    return {
        "puntaje_total": puntaje_total,
        "nivel_global": nivel_global,
        "resumen": f"Tu respuesta obtuvo {puntaje_total}% de coherencia semántica. El nivel global es {nivel_global}.",
        "similitud_caso": round(similitud_caso, 4),
        "similitud_contexto": round(similitud_contexto, 4),
        "indice_relevancia_lexica": round(indice_lexico, 4),
        "indice_relevancia_caso": round(indice_relevancia_caso, 4),
        "evidencia_principal": evidencia,
        "criterios": criterios,
        "observaciones": observaciones,
        "recomendaciones": recomendaciones,
        "palabras_clave_caso": palabras_clave,
    }