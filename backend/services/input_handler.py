#Limpia texto y extrae texto de un PDF
from pathlib import Path
from pypdf import PdfReader


def normalizar_texto(texto: str) -> str:
    if not texto:
        return ""
    return " ".join(texto.split()).strip()


def extraer_texto_pdf(ruta_pdf: str | Path) -> str:
    ruta = Path(ruta_pdf)

    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo PDF: {ruta}")

    lector = PdfReader(str(ruta))
    partes = []

    for pagina in lector.pages:
        partes.append(pagina.extract_text() or "")

    texto = "\n".join(partes)
    return normalizar_texto(texto)