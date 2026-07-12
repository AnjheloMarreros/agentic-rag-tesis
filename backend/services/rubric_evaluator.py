from __future__ import annotations

from typing import Any
from difflib import SequenceMatcher
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

STOPWORDS_CONCEPTUALES = STOPWORDS | {
    "esto", "esta", "este", "esas", "esos", "aquello", "aquella", "aquellas", "aquellos",
    "puede", "pueden", "debe", "deben", "ser", "hay", "como", "tambien", "también",
    "porque", "aunque", "mientras", "ademas", "además", "sin", "no", "si", "sí",
    "redacta", "incluye", "evita", "usa", "orden", "lógico", "logico", "conectores",
    "premisas", "conclusion", "conclusión", "sustento", "juridico", "jurídico",
    "argumentativos", "argumentativo", "contenido", "claros", "claro", "relacion",
    "relación", "foco",
}

GENERIC_INSTRUCTION_WORDS = {
    "redacta", "respuesta", "orden", "logico", "lógico", "incluye", "evita", "usa",
    "premisas", "conclusion", "conclusión", "sustento", "juridico", "jurídico",
    "argumentativos", "argumentativo", "claros", "claro", "foco", "hechos",
    "cuestion", "cuestión", "problema", "juridica", "jurídica", "caso", "casos",
    "texto", "argumenta", "argumentar", "relacion", "relación",
}

CASE_PROFILE_KEYS = (
    "perfil_juridico",
    "hechos_clave",
    "normas_clave",
    "conceptos_esperados",
    "tesis_esperada",
    "palabras_clave",
)

NGRAM_RANGES = (2, 3, 4)


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


def _es_unidad_valida(unidad: str) -> bool:
    unidad_n = _normalizar(unidad)
    if not unidad_n:
        return False

    tokens = unidad_n.split()
    if not tokens:
        return False

    if all(t in STOPWORDS_CONCEPTUALES or t in GENERIC_INSTRUCTION_WORDS for t in tokens):
        return False

    return any(
        len(t) >= 5 and t not in STOPWORDS_CONCEPTUALES and t not in GENERIC_INSTRUCTION_WORDS
        for t in tokens
    )


def _frases(texto: str, tamanios: tuple[int, ...] = NGRAM_RANGES) -> set[str]:
    tokens = _tokens(texto)
    frases: set[str] = set()
    for n in tamanios:
        if n <= 0 or len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            frase = " ".join(tokens[i : i + n]).strip()
            if frase and _es_unidad_valida(frase):
                frases.add(frase)
    return frases


def _peso_unidad(unidad: str) -> float:
    # Las frases jurídicas relevantes deben pesar más que una palabra suelta.
    if " " in unidad:
        n_tokens = len(unidad.split())
        if n_tokens >= 4:
            return 1.60
        if n_tokens == 3:
            return 1.40
        return 1.20
    return 1.00


def _coincidencia_fuzzy(unidad: str, referencia: str) -> bool:
    if unidad == referencia:
        return True

    if len(unidad) < 4 or len(referencia) < 4:
        return False

    if unidad in referencia or referencia in unidad:
        return True

    ratio = SequenceMatcher(None, unidad, referencia).ratio()
    return ratio >= 0.82


def _coincidencias_aproximadas(resp_units: set[str], ref_units: set[str]) -> set[str]:
    coincidencias: set[str] = set()

    ref_list = list(ref_units)
    for unidad in resp_units:
        if unidad in ref_units:
            coincidencias.add(unidad)
            continue

        for ref in ref_list:
            if _coincidencia_fuzzy(unidad, ref):
                coincidencias.add(unidad)
                break

    return coincidencias


def _texto_referencia(caso: Any, fuentes: Any) -> str:
    partes: list[str] = []

    if isinstance(caso, dict):
        # Se usa el contenido sustantivo del caso; las instrucciones genéricas no deben dominar el criterio.
        _agregar_texto(partes, caso.get("titulo", ""))
        _agregar_texto(partes, caso.get("enunciado", ""))
        _agregar_texto(partes, caso.get("contexto", []))

        perfil = caso.get("perfil_juridico")
        if isinstance(perfil, dict):
            for key in CASE_PROFILE_KEYS[1:]:
                _agregar_texto(partes, perfil.get(key, []))

    if isinstance(fuentes, list):
        for item in fuentes:
            if isinstance(item, dict):
                _agregar_texto(partes, item.get("fragmento", ""))

    return " ".join(partes).strip()


def _texto_caso_base(caso: Any) -> str:
    partes: list[str] = []

    if isinstance(caso, dict):
        _agregar_texto(partes, caso.get("titulo", ""))
        _agregar_texto(partes, caso.get("enunciado", ""))
        _agregar_texto(partes, caso.get("contexto", []))

        perfil = caso.get("perfil_juridico")
        if isinstance(perfil, dict):
            for key in CASE_PROFILE_KEYS[1:]:
                _agregar_texto(partes, perfil.get(key, []))

    return " ".join(partes).strip()


def _palabras_clave_caso(caso: dict[str, Any]) -> list[str]:
    textos: list[str] = []

    _agregar_texto(textos, caso.get("titulo", ""))
    _agregar_texto(textos, caso.get("enunciado", ""))
    _agregar_texto(textos, caso.get("contexto", []))

    perfil = caso.get("perfil_juridico")
    if isinstance(perfil, dict):
        for key in CASE_PROFILE_KEYS[1:]:
            _agregar_texto(textos, perfil.get(key, []))

    texto = " ".join(textos)
    tokens = _tokens(texto)
    claves = [
        tok for tok in tokens
        if len(tok) >= 5
        and tok not in STOPWORDS_CONCEPTUALES
        and tok not in GENERIC_INSTRUCTION_WORDS
    ]

    frases = _frases(texto)
    frases_filtradas = [frase for frase in frases if _es_unidad_valida(frase)]

    candidatos = list(dict.fromkeys(claves + frases_filtradas))
    return candidatos[:80]


def _extraer_unidades(texto: str) -> set[str]:
    texto_n = _normalizar(texto)
    tokens = _tokens(texto_n)

    unidades: set[str] = set()
    for tok in tokens:
        if len(tok) >= 5 and tok not in STOPWORDS_CONCEPTUALES and tok not in GENERIC_INSTRUCTION_WORDS:
            unidades.add(tok)

    for n in NGRAM_RANGES:
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            frase = " ".join(tokens[i : i + n]).strip()
            if frase and _es_unidad_valida(frase):
                unidades.add(frase)

    return unidades


def _indice_relevancia_lexica(respuesta: str, palabras_clave: list[str]) -> float:
    resp_units = _extraer_unidades(respuesta)
    keys = {k for k in palabras_clave if _es_unidad_valida(k)}

    if not keys or not resp_units:
        return 0.0

    exactas = resp_units.intersection(keys)
    aproximadas = _coincidencias_aproximadas(resp_units - exactas, keys - exactas)

    if not exactas and not aproximadas:
        return 0.0

    overlap = exactas | aproximadas

    peso_overlap = sum(_peso_unidad(u) for u in overlap)
    peso_resp = sum(_peso_unidad(u) for u in resp_units)
    peso_keys = sum(_peso_unidad(u) for u in keys)

    cobertura_clave = peso_overlap / max(1e-9, peso_keys)
    cobertura_respuesta = peso_overlap / max(1e-9, peso_resp)

    return (0.55 * cobertura_clave) + (0.45 * cobertura_respuesta)


def _cobertura_referencia(respuesta: str, texto_referencia: str) -> float:
    resp = _extraer_unidades(respuesta)
    ref = _extraer_unidades(texto_referencia)
    if not resp or not ref:
        return 0.0

    exactas = resp.intersection(ref)
    aproximadas = _coincidencias_aproximadas(resp - exactas, ref - exactas)

    if not exactas and not aproximadas:
        return 0.0

    overlap = exactas | aproximadas

    peso_overlap = sum(_peso_unidad(u) for u in overlap)
    peso_ref = sum(_peso_unidad(u) for u in ref)
    peso_resp = sum(_peso_unidad(u) for u in resp)

    cobertura_ref = peso_overlap / max(1e-9, peso_ref)
    cobertura_resp = peso_overlap / max(1e-9, peso_resp)

    return (0.60 * cobertura_ref) + (0.40 * cobertura_resp)


def _coincidencias_frasales(respuesta: str, texto_referencia: str) -> set[str]:
    resp_frases = _frases(respuesta)
    ref_frases = _frases(texto_referencia)
    if not resp_frases or not ref_frases:
        return set()

    exactas = resp_frases.intersection(ref_frases)
    aproximadas: set[str] = set()

    ref_list = list(ref_frases)
    for frase in resp_frases - exactas:
        for ref in ref_list:
            if _coincidencia_fuzzy(frase, ref):
                aproximadas.add(frase)
                break

    return exactas | aproximadas


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

    if not texto_n or not ref_texto:
        puntaje = 1
        return {
            "puntaje": puntaje,
            "nivel": _nivel(puntaje),
            "observacion": "No fue posible medir adecuadamente la pertinencia con el caso.",
            "recomendacion": "Vincula tu respuesta de forma más directa con el caso planteado.",
        }

    ref_units = _extraer_unidades(ref_texto)
    resp_units = _extraer_unidades(texto_n)

    if not ref_units or not resp_units:
        puntaje = 1
        return {
            "puntaje": puntaje,
            "nivel": _nivel(puntaje),
            "observacion": "No fue posible medir adecuadamente la pertinencia con el caso.",
            "recomendacion": "Vincula tu respuesta de forma más directa con el caso planteado.",
        }

    exactas = resp_units.intersection(ref_units)
    fuzzy = _coincidencias_aproximadas(resp_units - exactas, ref_units - exactas)
    frases = _coincidencias_frasales(texto_n, ref_texto)

    palabras_clave = _palabras_clave_caso(caso if isinstance(caso, dict) else {})
    indice_lexico = _indice_relevancia_lexica(texto_n, palabras_clave)

    similitud_caso = _cobertura_referencia(texto_n, _texto_caso_base(caso))
    similitud_contexto = _cobertura_referencia(texto_n, " ".join(
        str(item.get("fragmento", ""))
        for item in (fuentes or [])
        if isinstance(item, dict)
    ).strip())

    peso_exactas = sum(_peso_unidad(u) for u in exactas)
    peso_fuzzy = sum(_peso_unidad(u) for u in fuzzy)
    peso_frases = sum(_peso_unidad(u) for u in frases)

    peso_overlap = peso_exactas + (0.85 * peso_fuzzy) + (1.15 * peso_frases)
    peso_ref = sum(_peso_unidad(u) for u in ref_units)
    peso_resp = sum(_peso_unidad(u) for u in resp_units)

    cobertura_referencia = peso_overlap / max(1e-9, peso_ref)
    cobertura_respuesta = peso_overlap / max(1e-9, peso_resp)

    # Señal principal de pertinencia: unidades sustantivas del caso + recuperación contextual.
    score_ratio = (
        (0.40 * cobertura_referencia)
        + (0.25 * cobertura_respuesta)
        + (0.20 * indice_lexico)
        + (0.10 * similitud_caso)
        + (0.05 * similitud_contexto)
    )

    if score_ratio >= 0.34:
        puntaje = 5
    elif score_ratio >= 0.24:
        puntaje = 4
    elif score_ratio >= 0.14:
        puntaje = 3
    elif score_ratio >= 0.05:
        puntaje = 2
    else:
        puntaje = 1

    # Si hay evidencia real del caso, no puede caer en 1.
    evidencia_real = bool(exactas or fuzzy or frases or indice_lexico > 0 or similitud_caso > 0 or similitud_contexto > 0)
    if evidencia_real:
        puntaje = max(puntaje, 2)

    # Si la respuesta toca al menos dos núcleos sustantivos del caso, debe subir al menos a 3.
    if (len(exactas) + len(fuzzy) + len(frases)) >= 2:
        puntaje = max(puntaje, 3)

    observacion = (
        f"La respuesta mantiene {len(exactas)} coincidencias exactas, {len(fuzzy)} coincidencias aproximadas "
        f"y {len(frases)} coincidencias frasales relevantes con el caso."
        if evidencia_real
        else "La respuesta no presenta coincidencias relevantes con el caso."
    )

    recomendacion = (
        "La pertinencia con el caso es adecuada."
        if puntaje >= 4
        else "Relaciona más tu respuesta con los hechos, la norma y la tesis jurídica del caso."
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