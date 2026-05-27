import re


CONECTORES = [
    "porque",
    "por tanto",
    "sin embargo",
    "en consecuencia",
    "además",
    "por ello",
    "por consiguiente"
]


TERMINOS_JURIDICOS = [
    "derecho",
    "norma",
    "constitución",
    "proceso",
    "tribunal",
    "demanda",
    "identidad",
    "caducidad",
    "control difuso",
    "código civil"
]


def detectar_conectores(texto: str):

    texto = texto.lower()

    encontrados = [
        c for c in CONECTORES
        if c in texto
    ]

    return encontrados


def detectar_terminos_juridicos(texto: str):

    texto = texto.lower()

    encontrados = [
        t for t in TERMINOS_JURIDICOS
        if t in texto
    ]

    return encontrados


def calcular_estructura(texto: str):

    puntaje = 0
    observacion = []
    recomendacion = []

    longitud = len(texto.split())

    if longitud > 40:
        puntaje += 2
        observacion.append(
            "La respuesta tiene una extensión adecuada."
        )
    else:
        recomendacion.append(
            "Amplía la longitud de tu respuesta."
        )

    conectores = detectar_conectores(texto)

    if len(conectores) >= 2:
        puntaje += 2
        observacion.append(
            "Se identificaron conectores argumentativos."
        )
    else:
        recomendacion.append(
            "Usa más conectores argumentativos."
        )

    if "." in texto:
        puntaje += 1

    return {
        "puntaje": puntaje,
        "nivel": obtener_nivel(puntaje),
        "observacion": " ".join(observacion),
        "recomendacion": " ".join(recomendacion)
    }


def calcular_relevancia(texto: str):

    terminos = detectar_terminos_juridicos(texto)

    puntaje = min(len(terminos), 5)

    return {
        "puntaje": puntaje,
        "nivel": obtener_nivel(puntaje),
        "observacion": (
            f"Se encontraron {len(terminos)} términos jurídicos relevantes."
        ),
        "recomendacion": (
            "Relaciona más tu respuesta con fundamentos jurídicos."
            if puntaje < 3 else
            "La pertinencia jurídica es adecuada."
        )
    }


def calcular_consistencia(texto: str):

    puntaje = 3

    if "pero" in texto.lower() and "sin embargo" in texto.lower():
        puntaje += 1

    if len(texto.split()) > 80:
        puntaje += 1

    puntaje = min(puntaje, 5)

    return {
        "puntaje": puntaje,
        "nivel": obtener_nivel(puntaje),
        "observacion": (
            "La respuesta mantiene una relación lógica aceptable."
        ),
        "recomendacion": (
            "Refuerza la relación entre premisas y conclusión."
        )
    }


def calcular_cohesion(texto: str):

    longitud = len(texto.split())

    if longitud < 20:
        puntaje = 1
    elif longitud < 40:
        puntaje = 2
    elif longitud < 70:
        puntaje = 3
    elif longitud < 100:
        puntaje = 4
    else:
        puntaje = 5

    return {
        "puntaje": puntaje,
        "nivel": obtener_nivel(puntaje),
        "observacion": (
            "La cohesión textual fue evaluada según la organización del discurso."
        ),
        "recomendacion": (
            "Usa párrafos y conectores para mejorar la cohesión."
        )
    }


def obtener_nivel(puntaje: int):

    if puntaje <= 1:
        return "Muy bajo"

    if puntaje == 2:
        return "Bajo"

    if puntaje == 3:
        return "Medio"

    if puntaje == 4:
        return "Alto"

    return "Excelente"


def evaluar_respuesta_con_rubrica(
    respuesta,
    caso,
    fuentes,
    rubrica
):

    criterios = []

    total = 0

    estructura = calcular_estructura(respuesta)
    relevancia = calcular_relevancia(respuesta)
    consistencia = calcular_consistencia(respuesta)
    cohesion = calcular_cohesion(respuesta)

    evaluaciones = {
        "estructura_logica": estructura,
        "relevancia": relevancia,
        "consistencia": consistencia,
        "cohesion": cohesion
    }

    for criterio in rubrica["criterios"]:

        clave = criterio["clave"]

        resultado = evaluaciones[clave]

        puntaje = resultado["puntaje"]

        total += puntaje * criterio["peso"]

        criterios.append({
            "clave": clave,
            "nombre": criterio["nombre"],
            "peso": criterio["peso"],
            "puntaje": puntaje,
            "nivel": resultado["nivel"],
            "observacion": resultado["observacion"],
            "recomendacion": resultado["recomendacion"]
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

    recomendaciones_generales = []

    for c in criterios:
        if c["puntaje"] <= 3:
            recomendaciones_generales.append(
                c["recomendacion"]
            )

    return {
        "puntaje_total": porcentaje,
        "nivel_global": nivel_global,
        "resumen": (
            f"Tu respuesta obtuvo "
            f"{porcentaje}% de coherencia argumentativa. "
            f"El nivel global es {nivel_global}."
        ),
        "criterios": criterios,
        "recomendaciones_generales": recomendaciones_generales
    }