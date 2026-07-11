from __future__ import annotations

from typing import Any
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
    texto = (texto or "").lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.replace("¿", " ").replace("?", " ").replace("¡", " ").replace("!", " ")
    texto = texto.replace(".", " ").replace(",", " ").replace(";", " ").replace(":", " ")
    texto = texto.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
    texto = texto.replace("{", " ").replace("}", " ").replace("/", " ").replace("\\", " ")
    return " ".join(texto.split()).strip()


def _tokens(texto: str) -> list[str]:
    texto_n = _normalizar(texto)
    tokens = [t for t in texto_n.split() if len(t) > 2]
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


def _agregar_texto(partes: list[str], valor: Any) -> None:
    if valor is None:
        return
    if isinstance(valor, str):
        texto = valor.strip()
        if texto:
            partes.append(texto)
        return
    if isinstance(valor, list):
        for item in valor:
            if item is None:
                continue
            texto = str(item).strip()
            if texto:
                partes.append(texto)
        return
    texto = str(valor).strip()
    if texto:
        partes.append(texto)


def _texto_referencia(caso: Any, fuentes: Any) -> str:
    partes: list[str] = []

    if isinstance(caso, dict):
        _agregar_texto(partes, caso.get("titulo", ""))
        _agregar_texto(partes, caso.get("enunciado", ""))
        _agregar_texto(partes, caso.get("contexto", []))
        _agregar_texto(partes, caso.get("instrucciones", []))

    if isinstance(fuentes, list):
        for item in fuentes:
            if isinstance(item, dict):
                _agregar_texto(partes, item.get("fragmento", ""))

    return " ".join(partes).strip()


def _coincidencia_fuzzy(token: str, referencia: str) -> bool:
    if token == referencia:
        return True
    if len(token) >= 4 and len(referencia) >= 4:
        pref = 5 if len(token) >= 5 and len(referencia) >= 5 else 4
        if token[:pref] == referencia[:pref]:
            return True
        if token in referencia or referencia in token:
            return True
    return False


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

    if any(m in texto_n for m in MARCADORES_PREMISA):
        puntaje += 1
        observacion.append("La respuesta presenta estructura de premisas o fundamento.")
    else:
        recomendacion.append("Incluye premisas explícitas o un sustento inicial claro.")

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

    ref_tokens = list(dict.fromkeys(_tokens(ref_texto)))
    resp_tokens = list(dict.fromkeys(_tokens(texto_n)))

    if not ref_tokens or not resp_tokens:
        puntaje = 1
        return {
            "puntaje": puntaje,
            "nivel": _nivel(puntaje),
            "observacion": "No fue posible medir adecuadamente la pertinencia con el caso.",
            "recomendacion": "Vincula tu respuesta de forma más directa con el caso planteado.",
        }

    exactas = set(resp_tokens).intersection(ref_tokens)

    fuzzy: set[str] = set()
    for token in resp_tokens:
        if token in exactas:
            continue
        for ref in ref_tokens:
            if _coincidencia_fuzzy(token, ref):
                fuzzy.add(token)
                break

    coincidencias_ponderadas = len(exactas) + (0.6 * len(fuzzy))
    cobertura_referencia = coincidencias_ponderadas / max(1, len(ref_tokens))
    cobertura_respuesta = coincidencias_ponderadas / max(1, len(resp_tokens))
    score_ratio = (0.65 * cobertura_referencia) + (0.35 * cobertura_respuesta)

    if coincidencias_ponderadas <= 0:
        puntaje = 1
    elif score_ratio >= 0.30:
        puntaje = 5
    elif score_ratio >= 0.22:
        puntaje = 4
    elif score_ratio >= 0.14:
        puntaje = 3
    elif score_ratio >= 0.06:
        puntaje = 2
    else:
        puntaje = 1

    if coincidencias_ponderadas > 0:
        puntaje = max(puntaje, 2)
        if coincidencias_ponderadas >= 4:
            puntaje = max(puntaje, 3)

    observacion = (
        f"La respuesta mantiene {len(exactas)} coincidencias exactas y {len(fuzzy)} coincidencias aproximadas relevantes con el caso."
        if coincidencias_ponderadas > 0
        else "La respuesta no presenta coincidencias relevantes con el caso."
    )

    recomendacion = (
        "La pertinencia con el caso es adecuada."
        if puntaje >= 4
        else "Relaciona más tu respuesta con hechos, problema y sustento jurídico del caso."
    )

    return {
        "puntaje": puntaje,
        "nivel": _nivel(puntaje),
        "observacion": observacion,
        "recomendacion": recomendacion,
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
    criterios_rubrica = [
        c for c in criterios_rubrica
        if c.get("clave") != "relevancia_caso"
        and c.get("nombre") != "Relevancia con el caso"
    ]

    if not criterios_rubrica:
        criterios_rubrica = [
            {"clave": "estructura_logica", "nombre": "Estructura lógica del argumento", "peso": 0.25},
            {"clave": "relevancia", "nombre": "Relevancia y pertinencia del contenido", "peso": 0.25},
            {"clave": "consistencia", "nombre": "Consistencia lógica del razonamiento", "peso": 0.25},
            {"clave": "cohesion", "nombre": "Cohesión y calidad discursiva", "peso": 0.25},
        ]

    for criterio in criterios_rubrica:
        clave = criterio["clave"]
        resultado = evaluaciones.get(
            clave,
            {"puntaje": 1, "nivel": "Muy bajo", "observacion": "", "recomendacion": ""},
        )
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
        "puntaje_total_rubrica": porcentaje,
        "nivel_global": nivel_global,
        "resumen": (
            f"Tu respuesta obtuvo {porcentaje}% de coherencia argumentativa. "
            f"El nivel global es {nivel_global}."
        ),
        "criterios": criterios,
        "recomendaciones_generales": recomendaciones_generales,
    }