def generar_retroalimentacion(texto: str) -> dict:
    if not texto:
        return {
            "estado": "error",
            "observaciones": ["No se recibió texto para evaluar."],
            "recomendaciones": ["Escribe tu respuesta antes de enviarla."]
        }

    observaciones = []
    recomendaciones = []

    longitud = len(texto)

    if longitud < 150:
        observaciones.append("La respuesta es muy breve para una evaluación jurídica sólida.")
        recomendaciones.append("Amplía tu argumento con premisas, fundamento y conclusión.")
    else:
        observaciones.append("La respuesta tiene una extensión suficiente para una evaluación inicial.")
        recomendaciones.append("Ahora revisa si tus ideas están conectadas con lógica y coherencia.")

    texto_minuscula = texto.lower()

    if "porque" not in texto_minuscula and "por tanto" not in texto_minuscula and "sin embargo" not in texto_minuscula:
        observaciones.append("Faltan conectores argumentativos visibles.")
        recomendaciones.append("Usa conectores como 'porque', 'por tanto', 'sin embargo' o 'en consecuencia'.")

    return {
        "estado": "prototipo",
        "observaciones": observaciones,
        "recomendaciones": recomendaciones
    }