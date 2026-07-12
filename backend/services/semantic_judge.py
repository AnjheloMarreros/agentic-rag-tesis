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
    "asimismo",
    "de igual forma",
    "de igual manera",
    "en primer lugar",
    "en segundo lugar",
    "ahora bien",
    "en ese sentido",
    "cabe señalar",
    "cabe resaltar",
    "conviene precisar",
    "de este modo",
    "de esta forma",
    "en efecto",
    "en suma",
    "en síntesis",
    "en sintesis",
    "finalmente",
    "por ende",
    "por lo que",
    "a su vez",
    "sin perjuicio de ello",
    "sino también",
    "por lo demás",
    "en virtud de ello",
    "dicho esto",
    "en cambio",
    "de manera que",
    "esto es",
    "es decir",
    "por un lado",
    "por el otro",
    "en todo caso",
    "desde luego",
    "en definitiva",
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
    "sustento",
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

STOPWORDS_CONCEPTUALES = STOPWORDS | {
    "esto", "esta", "este", "esas", "esos", "aquello", "aquella", "aquellas", "aquellos",
    "puede", "pueden", "debe", "deben", "ser", "hay", "como", "tambien", "también",
    "porque", "aunque", "mientras", "ademas", "además", "sin", "no", "si", "sí",
    "redacta", "incluye", "evita", "usa", "orden", "lógico", "logico", "conectores",
    "premisas", "conclusion", "conclusión", "sustento", "juridico", "jurídico",
    "argumentativos", "argumentativo", "contenido",
}

CASE_PROFILE_KEYS = (
    "perfil_juridico",
    "hechos_clave",
    "normas_clave",
    "conceptos_esperados",
    "tesis_esperada",
    "palabras_clave",
)

# Instrucciones genéricas: no deben dominar el criterio de pertinencia.
GENERIC_INSTRUCTION_WORDS = {
    "redacta", "respuesta", "orden", "logico", "lógico", "incluye", "evita", "usa",
    "premisas", "conclusion", "conclusión", "sustento", "juridico", "jurídico",
    "argumentativos", "argumentativo", "claros", "claro", "foco", "hechos", "cuestion",
    "cuestión", "problema", "juridica", "jurídica", "caso", "casos", "texto",
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


def _frases(texto: str, tamanios: tuple[int, ...] = (2, 3)) -> set[str]:
    tokens = _tokens(texto)
    frases: set[str] = set()
    for n in tamanios:
        if n <= 0 or len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            frase = " ".join(tokens[i : i + n]).strip()
            if frase:
                frases.add(frase)
    return frases


def _texto_referencia(caso: dict[str, Any] | None, contexto_recuperado: list[dict[str, Any]]) -> str:
    partes: list[str] = []

    if isinstance(caso, dict):
        _agregar_texto(partes, caso.get("titulo", ""))
        _agregar_texto(partes, caso.get("enunciado", ""))
        _agregar_texto(partes, caso.get("contexto", []))

        perfil = caso.get("perfil_juridico")
        if isinstance(perfil, dict):
            for key in CASE_PROFILE_KEYS[1:]:
                _agregar_texto(partes, perfil.get(key, []))

    for item in contexto_recuperado:
        if isinstance(item, dict):
            _agregar_texto(partes, item.get("fragmento", ""))

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

        score = len(resp_tokens & frag_tokens) / max(1, len(frag_tokens))
        if score > mejor_score:
            mejor_score = score
            mejor_fragmento = fragmento

    return mejor_fragmento


def _es_unidad_valida(unidad: str) -> bool:
    unidad_n = _normalizar(unidad)
    if not unidad_n:
        return False

    tokens = unidad_n.split()
    if not tokens:
        return False

    # Evita que palabras completamente genéricas dominen la evaluación.
    if all(t in STOPWORDS_CONCEPTUALES or t in GENERIC_INSTRUCTION_WORDS for t in tokens):
        return False

    # Mantén unidades con al menos un término sustantivo.
    return any(
        (len(t) >= 5 and t not in STOPWORDS_CONCEPTUALES and t not in GENERIC_INSTRUCTION_WORDS)
        for t in tokens
    )


def _palabras_clave_caso(caso: dict[str, Any]) -> list[str]:
    textos: list[str] = []

    # Solo el contenido sustantivo del caso; las instrucciones suelen ser genéricas.
    _agregar_texto(textos, caso.get("titulo", ""))
    _agregar_texto(textos, caso.get("enunciado", ""))
    _agregar_texto(textos, caso.get("contexto", []))

    perfil = caso.get("perfil_juridico")
    if isinstance(perfil, dict):
        for key in CASE_PROFILE_KEYS[1:]:
            _agregar_texto(textos, perfil.get(key, []))

    texto = " ".join(textos)
    tokens = _tokens(texto)
    claves = [tok for tok in tokens if len(tok) >= 5 and tok not in STOPWORDS_CONCEPTUALES and tok not in GENERIC_INSTRUCTION_WORDS]

    frases = _frases(texto, tamanios=(2, 3))
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

    for n in (2, 3):
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            frase = " ".join(tokens[i : i + n]).strip()
            if frase and _es_unidad_valida(frase):
                unidades.add(frase)

    return unidades


def _peso_unidad(unidad: str) -> float:
    # Las frases cuentan más que las palabras sueltas.
    return 1.35 if " " in unidad else 1.0


def _indice_relevancia_lexica(respuesta: str, palabras_clave: list[str]) -> float:
    resp_units = _extraer_unidades(respuesta)
    keys = {k for k in palabras_clave if _es_unidad_valida(k)}

    if not keys or not resp_units:
        return 0.0

    overlap = resp_units.intersection(keys)
    if not overlap:
        return 0.0

    peso_overlap = sum(_peso_unidad(u) for u in overlap)
    peso_resp = sum(_peso_unidad(u) for u in resp_units)
    peso_keys = sum(_peso_unidad(u) for u in keys)

    cobertura_respuesta = peso_overlap / max(1e-9, peso_resp)
    cobertura_clave = peso_overlap / max(1e-9, peso_keys)

    return (0.50 * cobertura_clave) + (0.50 * cobertura_respuesta)


def _cobertura_referencia(respuesta: str, texto_referencia: str) -> float:
    resp = _extraer_unidades(respuesta)
    ref = _extraer_unidades(texto_referencia)
    if not resp or not ref:
        return 0.0

    overlap = resp.intersection(ref)
    if not overlap:
        return 0.0

    peso_overlap = sum(_peso_unidad(u) for u in overlap)
    peso_ref = sum(_peso_unidad(u) for u in ref)
    peso_resp = sum(_peso_unidad(u) for u in resp)

    cobertura_ref = peso_overlap / max(1e-9, peso_ref)
    cobertura_resp = peso_overlap / max(1e-9, peso_resp)
    return (0.60 * cobertura_ref) + (0.40 * cobertura_resp)


def _coincidencias_frasales(respuesta: str, texto_referencia: str) -> set[str]:
    resp_frases = _frases(respuesta, tamanios=(2, 3))
    ref_frases = _frases(texto_referencia, tamanios=(2, 3))
    return resp_frases.intersection(ref_frases)


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
        ]
    ).strip()

    texto_contexto = " ".join(
        str(item.get("fragmento", ""))
        for item in contexto_recuperado
        if isinstance(item, dict)
    ).strip()

    texto_referencia = " ".join(part for part in [texto_caso, texto_contexto] if part).strip()

    palabras_clave = _palabras_clave_caso(caso)
    indice_lexico = _indice_relevancia_lexica(respuesta, palabras_clave)

    similitud_caso = _cobertura_referencia(respuesta, texto_caso)
    similitud_contexto = _cobertura_referencia(respuesta, texto_contexto)

    frases_compartidas = _coincidencias_frasales(respuesta, texto_referencia)

    conceptos_resp = _extraer_unidades(respuesta)
    conceptos_ref = _extraer_unidades(texto_referencia)
    conceptos_compartidos = conceptos_resp.intersection(conceptos_ref)

    soporte_contextual = _cobertura_referencia(respuesta, texto_contexto) if texto_contexto else 0.0

    indice_relevancia_caso = (
        (0.35 * similitud_caso)
        + (0.25 * similitud_contexto)
        + (0.25 * indice_lexico)
        + (0.15 * soporte_contextual)
    )

    conectores = sum(1 for c in CONECTORES if c in texto_n)
    conclusion = _contiene_alguna(texto_n, MARCADORES_CONCLUSION)
    premisas = _contiene_alguna(texto_n, MARCADORES_PREMISA)
    desconocimiento = _contiene_alguna(texto_n, FRASES_DESCONOCIMIENTO)

    score_caso = 1 + (indice_relevancia_caso * 4.0)
    score_contexto = 1 + (max(similitud_contexto, indice_lexico, soporte_contextual) * 4.0)

    # Regla de seguridad: cualquier evidencia real sube al menos a 2.
    if (similitud_caso > 0 or indice_lexico > 0 or frases_compartidas or conceptos_compartidos) and score_caso < 2:
        score_caso = 2

    if len(conceptos_compartidos) >= 2 and score_caso < 3:
        score_caso = 3

    if len(conceptos_compartidos) >= 4 and score_caso < 4:
        score_caso = 4

    if frases_compartidas and score_caso < 3:
        score_caso = 3

    if conclusion and (similitud_caso > 0 or indice_lexico > 0 or frases_compartidas or conceptos_compartidos):
        score_caso = max(score_caso, 3)

    if (similitud_contexto > 0 or indice_lexico > 0 or soporte_contextual > 0) and score_contexto < 2:
        score_contexto = 2

    if len(frases_compartidas) >= 1 and score_contexto < 3:
        score_contexto = 3

    if conclusion:
        score_caso += 0.20
        score_contexto += 0.15

    score_caso = _clamp(score_caso)
    score_contexto = _clamp(score_contexto)

    if score_caso >= 4:
        obs_caso = "La respuesta guarda relación semántica con el caso."
        rec_caso = "La respuesta está bien alineada con el caso."
    elif score_caso == 3:
        obs_caso = "La respuesta se relaciona parcialmente con el caso."
        rec_caso = "Relaciona más los hechos y la norma aplicable."
    else:
        obs_caso = "La relación con el caso todavía es limitada."
        rec_caso = "Mantén el foco en el problema jurídico planteado."

    if score_contexto >= 4:
        obs_contexto = "La respuesta se apoya en parte del contexto recuperado."
        rec_contexto = "El sustento contextual es adecuado."
    elif score_contexto == 3:
        obs_contexto = "La respuesta usa parcialmente el contexto recuperado."
        rec_contexto = "Vincula explícitamente tus ideas con el contexto jurídico recuperado."
    else:
        obs_contexto = "La respuesta usa poco el contexto recuperado."
        rec_contexto = "Aprovecha mejor el contexto recuperado para sustentar tu postura."

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
    if score_caso <= 2 and score_contexto <= 2 and n_palabras >= 20:
        inconsistencia_penalizacion += 1

    score_consistencia = 3 + (similitud_caso * 0.9) + (similitud_contexto * 0.7) + (indice_lexico * 0.4) - inconsistencia_penalizacion
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