from __future__ import annotations

from typing import Any


CONECTORES = [
    "porque",
    "por tanto",
    "sin embargo",
    "en consecuencia",
    "además",
    "por ello",
    "por consiguiente",
    "por lo tanto",
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

STOPWORDS = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "u",
    "a", "ante", "bajo", "con", "contra", "desde", "durante", "entre", "hacia",
    "hasta", "para", "por", "segun", "según", "sin", "sobre", "tras", "del",
    "al", "que", "se", "es", "son", "ser", "como", "más", "menos", "muy",
    "en", "su", "sus", "le", "les", "lo", "ya", "no", "sí", "también",
    "caso", "casos", "respuesta", "argumento", "argumentar", "texto",
}


def _normalizar(texto: str) -> str:
    return " ".join((texto or "").lower().split()).strip()


def _tokens(texto: str) -> list[str]:
    texto_n = _normalizar(texto)
    tokens = [t for t in texto_n.replace(".", " ").replace(",", " ").split() if len(t) > 2]
    return [t for t in tokens if t not in STOPWORDS]


def detectar_conectores(texto: str) -> list[str]:
    texto_n = _normalizar(texto)
    return [c for c in CONECTORES if c in texto_n]


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


def _texto_referencia(caso: Any, fuentes: Any) -> str:
    partes: list[str] = []

    if isinstance(caso, dict):
        partes.append(str(caso.get("titulo", "")))
        partes.append(str(caso.get("enunciado", "")))
        partes.append(" ".join(caso.get("contexto", []) or []))
        partes.append(" ".join(caso.get("instrucciones", []) or []))

    if isinstance(fuentes, list):
        for item in fuentes:
            if isinstance(item, dict):
                partes.append(str(item.get("fragmento", "")))

    return " ".join(partes).strip()


def calcular_estructura(texto: str):
    texto_n = _normalizar(texto)
    tokens = _tokens(texto)
    conectores = detectar_conectores(texto_n)

    puntaje = 0
    observacion = []
    recomendacion = []

    if len(tokens) >= 45:
        puntaje += 2
        observacion.append("La respuesta tiene una extensión suficiente.")
    elif len(tokens) >= 25:
        puntaje += 1
        observacion.append("La respuesta tiene una extensión aceptable.")
    else:
        recomendacion.append("Amplía tu respuesta con más desarrollo argumentativo.")

    if len(conectores) >= 2:
        puntaje += 2
        observacion.append("Se identificaron conectores argumentativos.")
    elif len(conectores) == 1:
        puntaje += 1
        observacion.append("Se identificó al menos un conector argumentativo.")
    else:
        recomendacion.append("Usa más conectores argumentativos.")

    if any(m in texto_n for m in MARCADORES_CONCLUSION):
        puntaje += 1
        observacion.append("La respuesta incluye una conclusión explícita.")
    else:
        recomendacion.append("Incluye una conclusión clara y explícita.")

    puntaje = min(puntaje, 5)

    return {
        "puntaje": puntaje,
        "nivel": _nivel(puntaje),
        "observacion": " ".join(observacion).strip(),
        "recomendacion": " ".join(recomendacion).strip(),
    }


def calcular_relevancia(texto: str, caso: Any = None, fuentes: Any = None):
    texto_n = _normalizar(texto)
    ref_texto = _texto_referencia(caso, fuentes)
    ref_tokens = set(_tokens(ref_texto))
    resp_tokens = set(_tokens(texto_n))

    if not ref_tokens or not resp_tokens:
        puntaje = 1
        return {
            "puntaje": puntaje,
            "nivel": _nivel(puntaje),
            "observacion": "No fue posible medir adecuadamente la pertinencia con el caso.",
            "recomendacion": "Vincula tu respuesta de forma más directa con el caso planteado.",
        }

    coincidencias = resp_tokens.intersection(ref_tokens)
    n = len(coincidencias)

    if n >= 12:
        puntaje = 5
    elif n >= 8:
        puntaje = 4
    elif n >= 5:
        puntaje = 3
    elif n >= 2:
        puntaje = 2
    else:
        puntaje = 1

    return {
        "puntaje": puntaje,
        "nivel": _nivel(puntaje),
        "observacion": f"La respuesta mantiene {n} coincidencias relevantes con el caso.",
        "recomendacion": (
            "La pertinencia con el caso es adecuada."
            if puntaje >= 4
            else "Relaciona más tu respuesta con el problema jurídico concreto."
        ),
    }


def calcular_consistencia(texto: str):
    texto_n = _normalizar(texto)
    tokens = _tokens(texto_n)
    conectores = detectar_conectores(texto_n)

    puntaje = 3

    if len(tokens) < 20:
        puntaje = 1
    elif len(tokens) < 40:
        puntaje = 2
    elif len(tokens) < 70:
        puntaje = 3
    elif len(tokens) < 100:
        puntaje = 4
    else:
        puntaje = 5

    if len(conectores) >= 2 and any(m in texto_n for m in MARCADORES_CONCLUSION):
        puntaje = min(5, puntaje + 1)

    return {
        "puntaje": puntaje,
        "nivel": _nivel(puntaje),
        "observacion": "La respuesta mantiene una relación lógica aceptable.",
        "recomendacion": "Refuerza la relación entre premisas y conclusión.",
    }


def calcular_cohesion(texto: str):
    texto_n = _normalizar(texto)
    tokens = _tokens(texto_n)
    conectores = detectar_conectores(texto_n)

    if len(tokens) < 20:
        puntaje = 1
    elif len(tokens) < 40:
        puntaje = 2
    elif len(tokens) < 70:
        puntaje = 3
    elif len(tokens) < 100:
        puntaje = 4
    else:
        puntaje = 5

    if len(conectores) >= 2:
        puntaje = min(5, puntaje + 1)

    return {
        "puntaje": puntaje,
        "nivel": _nivel(puntaje),
        "observacion": "La cohesión textual fue evaluada según la organización del discurso.",
        "recomendacion": "Usa párrafos y conectores para mejorar la cohesión.",
    }


def evaluar_respuesta_con_rubrica(
    respuesta,
    caso,
    fuentes,
    rubrica,
):
    estructura = calcular_estructura(respuesta)
    relevancia = calcular_relevancia(respuesta, caso=caso, fuentes=fuentes)
    consistencia = calcular_consistencia(respuesta)
    cohesion = calcular_cohesion(respuesta)

    evaluaciones = {
        "estructura_logica": estructura,
        "relevancia": relevancia,
        "consistencia": consistencia,
        "cohesion": cohesion,
    }

    criterios = []
    total = 0.0

    criterios_rubrica = rubrica.get("criterios", []) if isinstance(rubrica, dict) else []
    if not criterios_rubrica:
        criterios_rubrica = [
            {"clave": "estructura_logica", "nombre": "Estructura lógica del argumento", "peso": 0.25},
            {"clave": "relevancia", "nombre": "Relevancia y pertinencia del contenido", "peso": 0.25},
            {"clave": "consistencia", "nombre": "Consistencia lógica del razonamiento", "peso": 0.25},
            {"clave": "cohesion", "nombre": "Cohesión y calidad discursiva", "peso": 0.25},
        ]

    for criterio in criterios_rubrica:
        clave = criterio["clave"]
        resultado = evaluaciones.get(clave, {"puntaje": 1, "nivel": "Muy bajo", "observacion": "", "recomendacion": ""})
        puntaje = resultado["puntaje"]
        peso = criterio.get("peso", 0.25)

        total += puntaje * peso

        criterios.append({
            "clave": clave,
            "nombre": criterio.get("nombre", clave),
            "peso": peso,
            "puntaje": puntaje,
            "nivel": resultado["nivel"],
            "observacion": resultado["observacion"],
            "recomendacion": resultado["recomendacion"],
        })

    porcentaje = round((total / 5) * 100, 2)

    if porcentaje >= 80:
        nivel_global = "Excelente"
    elif porcentaje >= 60:
        nivel_global = "Alto"
    elif porcentaje >= 40:
        nivel_global = "Medio"
    else:
        nivel_global = "Bajo"

    recomendaciones_generales = [
        c["recomendacion"] for c in criterios if c["puntaje"] <= 3 and c["recomendacion"]
    ]

    return {
        "puntaje_total": porcentaje,
        "nivel_global": nivel_global,
        "resumen": (
            f"Tu respuesta obtuvo {porcentaje}% de coherencia argumentativa. "
            f"El nivel global es {nivel_global}."
        ),
        "criterios": criterios,
        "recomendaciones_generales": recomendaciones_generales,
    }