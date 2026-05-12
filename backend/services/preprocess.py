from pypdf import PdfReader


def normalizar_texto(texto: str) -> str:
    return " ".join(texto.split()).strip()


def extraer_texto_pdf(ruta_pdf: str) -> str:
    lector = PdfReader(ruta_pdf)
    partes = []

    for pagina in lector.pages:
        texto = pagina.extract_text() or ""
        partes.append(texto)

    return normalizar_texto("\n".join(partes))


def transcribir_audio(ruta_audio: str) -> str:
    raise NotImplementedError(
        "La transcripción de audio se implementará en una fase posterior."
    )