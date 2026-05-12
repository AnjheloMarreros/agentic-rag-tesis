from pathlib import Path
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = BASE_DIR / "data" / "docs"


def leer_txt(ruta: Path) -> str:
    with open(ruta, "r", encoding="utf-8") as archivo:
        return archivo.read()


def leer_pdf(ruta: Path) -> str:
    lector = PdfReader(str(ruta))
    partes = []

    for pagina in lector.pages:
        partes.append(pagina.extract_text() or "")

    return "\n".join(partes)


def cargar_documentos():
    documentos = []

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de documentos: {DOCS_DIR}"
        )

    for archivo in DOCS_DIR.iterdir():
        if archivo.is_file():
            if archivo.suffix.lower() == ".txt":
                documentos.append({
                    "source": archivo.stem,
                    "type": "txt",
                    "text": leer_txt(archivo)
                })
            elif archivo.suffix.lower() == ".pdf":
                documentos.append({
                    "source": archivo.stem,
                    "type": "pdf",
                    "text": leer_pdf(archivo)
                })

    if not documentos:
        raise ValueError("No se encontraron documentos para cargar en la base vectorial.")

    return documentos