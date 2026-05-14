from functools import lru_cache
from pathlib import Path

import whisper

from backend.services.preprocess import normalizar_texto


@lru_cache(maxsize=1)
def get_whisper_model():
    return whisper.load_model("base")


def transcribir_audio(ruta_audio: str) -> str:
    ruta = Path(ruta_audio)

    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de audio: {ruta}")

    model = get_whisper_model()
    resultado = model.transcribe(str(ruta), fp16=False)

    return normalizar_texto(resultado.get("text", ""))